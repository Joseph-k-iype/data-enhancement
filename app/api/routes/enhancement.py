"""
Optimized enhancement API routes with improved request handling.
Replace app/api/routes/enhancement.py with this implementation.
"""

import json
import logging
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_openai import AzureChatOpenAI
from app.core.models import (
    DataElement, 
    EnhancedDataElement, 
    EnhancementRequest, 
    EnhancementResponse, 
    EnhancementStatus,
    ValidationResult,
    DataQualityStatus
)
from app.core.in_memory_job_store import job_store
from app.agents.workflow import OptimizedDataEnhancementWorkflow
from app.config.settings import get_llm
from app.utils.cache import cache_manager
import orjson  # Using orjson for faster JSON serialization

router = APIRouter(prefix="/api/v1", tags=["data-enhancement"])

logger = logging.getLogger(__name__)

# Initialize performance metrics
request_times = {}
# Queue to track in-progress request IDs
active_requests = set()
# Semaphore to limit concurrent background tasks
background_semaphore = asyncio.Semaphore(10)  # Allow up to 10 concurrent background tasks

def get_workflow() -> OptimizedDataEnhancementWorkflow:
    """
    Get the data enhancement workflow with caching.
    
    Returns:
        OptimizedDataEnhancementWorkflow: The enhancement workflow
    """
    # Cache the workflow instance per LLM to reduce initialization overhead
    llm = get_llm()
    llm_id = id(llm)
    
    # Check if workflow exists in module-level cache
    if not hasattr(get_workflow, '_workflow_cache'):
        get_workflow._workflow_cache = {}
    
    if llm_id not in get_workflow._workflow_cache:
        workflow = OptimizedDataEnhancementWorkflow(llm)
        get_workflow._workflow_cache[llm_id] = workflow
        logger.info(f"Created new workflow instance for LLM {llm_id}")
        
    return get_workflow._workflow_cache[llm_id]

@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics():
    """
    Get performance metrics for the API.
    
    Returns:
        Dict with performance metrics
    """
    metrics = {}
    
    for endpoint, times in request_times.items():
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            p95_time = sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max_time
            
            metrics[endpoint] = {
                "avg_time": avg_time,
                "max_time": max_time,
                "min_time": min_time,
                "p95_time": p95_time,
                "request_count": len(times)
            }
    
    active_count = len(active_requests)
    
    return {
        "metrics": metrics,
        "total_endpoints": len(metrics),
        "active_requests": active_count,
        "cache_stats": cache_manager.get_stats()
    }

# Helper function to track request time
def track_request_time(endpoint: str, time_taken: float):
    """
    Track request time for an endpoint.
    
    Args:
        endpoint: API endpoint path
        time_taken: Time taken to process the request in seconds
    """
    if endpoint not in request_times:
        request_times[endpoint] = []
    
    request_times[endpoint].append(time_taken)
    
    # Keep only the last 100 requests per endpoint
    if len(request_times[endpoint]) > 100:
        request_times[endpoint] = request_times[endpoint][-100:]

@router.post("/validate", response_model=Dict[str, Any])
async def validate_data_element(data_element: DataElement, request: Request):
    """
    Validate a data element against ISO/IEC 11179 standards.
    
    This endpoint performs initial validation without enhancement.
    
    Args:
        data_element: The data element to validate
        
    Returns:
        Dict with validation results in JSON format
    """
    request_id = f"validate_{uuid.uuid4().hex}"
    active_requests.add(request_id)
    start_time = time.time()
    
    try:
        logger.info(f"Validating data element: {data_element.id}")
        workflow = get_workflow()
        
        # Use caching for validation results
        cache_key = f"validation_{data_element.id}_{hash(data_element.existing_name)}_{hash(data_element.existing_description)}"
        cached_result = await cache_manager.async_get_job(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for validation of {data_element.id}")
            result = cached_result
        else:
            # Run just the validation step
            result = await workflow.validator.validate(data_element)
            
            # Cache the result for future requests
            await cache_manager.async_store_job(cache_key, "validation", "completed", result.dict())
        
        # Extract name and description feedback
        name_feedback = ""
        desc_feedback = ""
        if result.feedback:
            feedback_parts = result.feedback.split("\n\n")
            if len(feedback_parts) >= 1:
                name_feedback = feedback_parts[0].replace("Name feedback:", "").strip()
            if len(feedback_parts) >= 2:
                desc_feedback = feedback_parts[1].replace("Description feedback:", "").strip()
        
        # Extract separate name and description validity from feedback
        # By default, both follow the overall is_valid flag
        name_valid = result.is_valid
        desc_valid = result.is_valid
        
        # Check for specific invalid indicators in the feedback
        if "name validation failed" in result.feedback.lower() or "invalid name" in result.feedback.lower():
            name_valid = False
        if "description validation failed" in result.feedback.lower() or "invalid description" in result.feedback.lower():
            desc_valid = False
        
        # Track request time
        track_request_time(request.url.path, time.time() - start_time)
            
        # Use orjson for faster serialization
        response_data = {
            "id": data_element.id,
            "name_valid": name_valid,
            "name_feedback": name_feedback or "No specific feedback provided for name",
            "description_valid": desc_valid, 
            "description_feedback": desc_feedback or "No specific feedback provided for description",
            "quality_status": result.quality_status.value,
            "suggested_improvements": result.suggested_improvements
        }
        
        active_requests.remove(request_id)
        return response_data
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        
        # Track request time for errors too
        track_request_time(request.url.path, time.time() - start_time)
        
        active_requests.remove(request_id)
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

@router.post("/enhance", response_model=EnhancementResponse)
async def enhance_data_element(
    request: EnhancementRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    """
    Enhance a data element to meet ISO/IEC 11179 standards.
    This is an asynchronous operation that will run in the background.
    
    Args:
        request: Enhancement request with data element
        background_tasks: FastAPI background tasks
        
    Returns:
        EnhancementResponse with request ID and status
    """
    request_id = f"enhance_{uuid.uuid4().hex}"
    active_requests.add(request_id)
    start_time = time.time()
    
    # Use the provided ID as the request ID for tracking
    job_id = request.data_element.id
    logger.info(f"Enhancement request received for data element: {job_id}")
    
    # Get job from in-memory store
    job_data = job_store.get_job(job_id)
    
    try:
        if job_data:
            # Job exists, get status
            status = EnhancementStatus(job_data["status"])
            
            # If already completed or failed, return the result
            if status in [EnhancementStatus.COMPLETED, EnhancementStatus.FAILED]:
                response = EnhancementResponse(
                    request_id=job_id,
                    status=status,
                    enhanced_data=job_data["data"].get("result"),
                    error_message=job_data["data"].get("error")
                )
                
                # Track request time
                track_request_time(http_request.url.path, time.time() - start_time)
                
                active_requests.remove(request_id)
                return response
            
            # Otherwise, return the current status
            response = EnhancementResponse(
                request_id=job_id,
                status=status,
                enhanced_data=None,
                error_message=None
            )
            
            # Track request time
            track_request_time(http_request.url.path, time.time() - start_time)
            
            active_requests.remove(request_id)
            return response
        
        # Initialize job in memory
        job_store.store_job(
            job_id=job_id,
            job_type="enhancement",
            status=EnhancementStatus.PENDING.value,
            data={
                "request": request.dict(),
                "result": None,
                "error": None
            }
        )
        
        # Add the enhancement task to the background tasks with semaphore
        background_tasks.add_task(
            run_enhancement_job_with_semaphore,
            request_id=job_id,
            data_element=request.data_element,
            max_iterations=request.max_iterations
        )
        
        response = EnhancementResponse(
            request_id=job_id,
            status=EnhancementStatus.PENDING,
            enhanced_data=None,
            error_message=None
        )
        
        # Track request time
        track_request_time(http_request.url.path, time.time() - start_time)
        
        active_requests.remove(request_id)
        return response
        
    except Exception as e:
        logger.error(f"Error in enhancement request: {e}")
        
        # Track request time for errors too
        track_request_time(http_request.url.path, time.time() - start_time)
        
        active_requests.remove(request_id)
        raise HTTPException(status_code=500, detail=f"Enhancement error: {str(e)}")

@router.post("/enhance/stream", response_class=StreamingResponse)
async def stream_enhance_data_element(
    request: EnhancementRequest,
    http_request: Request,
):
    """
    Enhance a data element and stream the results as they become available.
    
    Args:
        request: Enhancement request with data element
        
    Returns:
        StreamingResponse with enhancement updates
    """
    request_id = f"stream_{uuid.uuid4().hex}"
    active_requests.add(request_id)
    start_time = time.time()
    
    async def enhancement_stream():
        try:
            workflow = get_workflow()
            
            # Stream the enhancement results
            async for result in workflow.stream_run(request.data_element, request.max_iterations):
                # Use orjson for faster serialization
                yield orjson.dumps(result) + b"\n"
                
            # Track request time at the end
            track_request_time(http_request.url.path, time.time() - start_time)
            active_requests.remove(request_id)
            
        except Exception as e:
            logger.error(f"Error in streaming enhancement: {e}")
            yield orjson.dumps({
                "status": "error",
                "message": str(e)
            }) + b"\n"
            
            # Track request time for errors too
            track_request_time(http_request.url.path, time.time() - start_time)
            active_requests.remove(request_id)
    
    return StreamingResponse(
        enhancement_stream(),
        media_type="application/x-ndjson"
    )

@router.get("/enhance/{request_id}", response_model=EnhancementResponse)
async def get_enhancement_status(request_id: str, request: Request):
    """
    Get the status of an enhancement job.
    
    Args:
        request_id: ID of the enhancement job
        
    Returns:
        EnhancementResponse with current status and results if available
    """
    tracking_id = f"status_{uuid.uuid4().hex}"
    active_requests.add(tracking_id)
    start_time = time.time()
    
    try:
        # Get job from in-memory store
        job_data = job_store.get_job(request_id)
        
        if not job_data:
            # Track request time for errors too
            track_request_time(request.url.path, time.time() - start_time)
            active_requests.remove(tracking_id)
            raise HTTPException(status_code=404, detail=f"Enhancement job {request_id} not found")
        
        # Convert status string to enum
        status = EnhancementStatus(job_data["status"])
        
        response = EnhancementResponse(
            request_id=request_id,
            status=status,
            enhanced_data=job_data["data"].get("result"),
            error_message=job_data["data"].get("error")
        )
        
        # Track request time
        track_request_time(request.url.path, time.time() - start_time)
        
        active_requests.remove(tracking_id)
        return response
    
    except Exception as e:
        if isinstance(e, HTTPException):
            active_requests.remove(tracking_id)
            raise e
        
        # Track request time for errors too
        track_request_time(request.url.path, time.time() - start_time)
        
        logger.error(f"Error getting enhancement status: {e}")
        active_requests.remove(tracking_id)
        raise HTTPException(status_code=500, detail=f"Error retrieving enhancement status: {str(e)}")

@router.post("/enhance/batch", response_model=List[str])
async def batch_enhance_data_elements(
    requests: List[EnhancementRequest],
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Enhance multiple data elements in batch mode.
    Returns a list of request IDs that can be used to check status.
    
    Args:
        requests: List of enhancement requests
        background_tasks: FastAPI background tasks
        
    Returns:
        List of request IDs
    """
    request_id = f"batch_{uuid.uuid4().hex}"
    active_requests.add(request_id)
    start_time = time.time()
    
    request_ids = []
    
    # Prepare batch job storage for better performance
    batch_jobs = []
    
    for req in requests:
        request_id = req.data_element.id
        request_ids.append(request_id)
        
        # Get job from in-memory store
        job_data = job_store.get_job(request_id)
        
        if job_data:
            # Job exists, get status
            status = EnhancementStatus(job_data["status"])
            
            # Skip if already completed or failed or in progress
            if status in [EnhancementStatus.COMPLETED, EnhancementStatus.FAILED, EnhancementStatus.IN_PROGRESS]:
                continue
        
        # Add to batch for bulk storage
        batch_jobs.append((
            request_id,
            "enhancement",
            EnhancementStatus.PENDING.value,
            {
                "request": req.dict(),
                "result": None,
                "error": None
            }
        ))
        
        # Add the enhancement task to the background tasks - limit to 5 concurrent tasks
        background_tasks.add_task(
            run_enhancement_job_with_semaphore,
            request_id=request_id,
            data_element=req.data_element,
            max_iterations=req.max_iterations
        )
    
    # Bulk store jobs for better performance
    if batch_jobs:
        job_store.bulk_store_jobs(batch_jobs)
    
    # Track request time
    track_request_time(request.url.path, time.time() - start_time)
    
    active_requests.remove(f"batch_{uuid.uuid4().hex}")
    return request_ids

@router.delete("/enhance/{request_id}", response_model=Dict[str, Any])
async def delete_enhancement_job(request_id: str, request: Request):
    """
    Delete an enhancement job from the system.
    
    Args:
        request_id: ID of the job to delete
        
    Returns:
        Dict with deletion message
    """
    tracking_id = f"delete_{uuid.uuid4().hex}"
    active_requests.add(tracking_id)
    start_time = time.time()
    
    try:
        # Get job from in-memory store
        job_data = job_store.get_job(request_id)
        
        if not job_data:
            # Track request time for errors too
            track_request_time(request.url.path, time.time() - start_time)
            active_requests.remove(tracking_id)
            raise HTTPException(status_code=404, detail=f"Enhancement job {request_id} not found")
        
        # Don't allow deleting running jobs
        if job_data["status"] == EnhancementStatus.IN_PROGRESS.value:
            # Track request time for errors too
            track_request_time(request.url.path, time.time() - start_time)
            active_requests.remove(tracking_id)
            raise HTTPException(status_code=400, detail=f"Cannot delete a job that is currently in progress")
        
        # Delete job
        job_store.delete_job(request_id)
        
        response = {"message": f"Enhancement job {request_id} deleted successfully"}
        
        # Track request time
        track_request_time(request.url.path, time.time() - start_time)
        
        active_requests.remove(tracking_id)
        return response
    
    except Exception as e:
        if isinstance(e, HTTPException):
            active_requests.remove(tracking_id)
            raise e
        
        # Track request time for errors too
        track_request_time(request.url.path, time.time() - start_time)
        
        logger.error(f"Error deleting enhancement job: {e}")
        active_requests.remove(tracking_id)
        raise HTTPException(status_code=500, detail=f"Error deleting enhancement job: {str(e)}")

@router.get("/system/active-requests", response_model=Dict[str, Any])
async def get_active_requests():
    """Get count of active requests."""
    return {
        "active_requests": len(active_requests),
        "request_ids": list(active_requests)
    }

@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_cache():
    """Clear the cache."""
    cache_manager.clear()
    return {"message": "Cache cleared successfully"}

async def run_enhancement_job_with_semaphore(request_id: str, data_element: DataElement, max_iterations: int = 2):
    """
    Run the enhancement job in the background with a semaphore to limit concurrency.
    
    Args:
        request_id: ID of the enhancement job
        data_element: The data element to enhance
        max_iterations: Maximum number of enhancement iterations
    """
    async with background_semaphore:
        await run_enhancement_job(request_id, data_element, max_iterations)

async def run_enhancement_job(request_id: str, data_element: DataElement, max_iterations: int = 2):
    """
    Run the enhancement job in the background.
    
    Args:
        request_id: ID of the enhancement job
        data_element: The data element to enhance
        max_iterations: Maximum number of enhancement iterations (reduced to 2 for better performance)
    """
    logger.info(f"Starting enhancement job for {request_id}")
    workflow = get_workflow()
    
    try:
        # Update job status to in progress
        await job_store.async_store_job(
            job_id=request_id,
            job_type="enhancement",
            status=EnhancementStatus.IN_PROGRESS.value,
            data=(await job_store.async_get_job(request_id))["data"]
        )
        
        # Run the workflow
        result = await workflow.run(data_element, max_iterations)
        
        # Get current job data
        job_data = (await job_store.async_get_job(request_id))["data"]
        
        # Update job status to completed
        await job_store.async_store_job(
            job_id=request_id,
            job_type="enhancement",
            status=EnhancementStatus.COMPLETED.value,
            data={
                "request": job_data["request"],
                "result": result.dict(),
                "error": None
            }
        )
        
        logger.info(f"Enhancement job completed for {request_id}")
        
    except Exception as e:
        # Update job status to failed
        logger.error(f"Enhancement job failed for {request_id}: {str(e)}")
        
        # Get current job data
        job_data = (await job_store.async_get_job(request_id))["data"]
        
        # Update job status to failed
        await job_store.async_store_job(
            job_id=request_id,
            job_type="enhancement",
            status=EnhancementStatus.FAILED.value,
            data={
                "request": job_data["request"],
                "result": None,
                "error": str(e)
            }
        )