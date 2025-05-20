"""
Enhancement API - Routes for data element enhancement.

This module provides API endpoints for enhancing data elements to meet ISO/IEC 11179
standards, with support for asynchronous processing, job tracking, and streaming responses.
"""

import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse
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

router = APIRouter(prefix="/api/v1", tags=["data-enhancement"])

logger = logging.getLogger(__name__)

# Initialize performance metrics
request_times = {}

def get_workflow() -> OptimizedDataEnhancementWorkflow:
    """
    Get the data enhancement workflow.
    
    Returns:
        OptimizedDataEnhancementWorkflow: The enhancement workflow
    """
    llm = get_llm()
    return OptimizedDataEnhancementWorkflow(llm)

@router.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time header to responses and collect metrics.
    """
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Track performance by endpoint
    endpoint = request.url.path
    if endpoint not in request_times:
        request_times[endpoint] = []
    
    request_times[endpoint].append(process_time)
    
    # Keep only the last 100 requests per endpoint
    if len(request_times[endpoint]) > 100:
        request_times[endpoint] = request_times[endpoint][-100:]
    
    return response

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
    
    return {
        "metrics": metrics,
        "total_endpoints": len(metrics)
    }

@router.post("/validate", response_model=Dict[str, Any])
async def validate_data_element(data_element: DataElement):
    """
    Validate a data element against ISO/IEC 11179 standards.
    
    This endpoint performs initial validation without enhancement.
    
    Args:
        data_element: The data element to validate
        
    Returns:
        Dict with validation results in JSON format
    """
    logger.info(f"Validating data element: {data_element.id}")
    workflow = get_workflow()
    try:
        # Run just the validation step
        result = await workflow.validator.validate(data_element)
        
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
            
        # Format as JSON with all 6 evaluation points, ensuring both name and description have feedback
        return {
            "id": data_element.id,
            "name_valid": name_valid,
            "name_feedback": name_feedback or "No specific feedback provided for name",
            "description_valid": desc_valid, 
            "description_feedback": desc_feedback or "No specific feedback provided for description",
            "quality_status": result.quality_status.value,
            "suggested_improvements": result.suggested_improvements
        }
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

@router.post("/enhance", response_model=EnhancementResponse)
async def enhance_data_element(
    request: EnhancementRequest,
    background_tasks: BackgroundTasks,
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
    # Use the provided ID as the request ID for tracking
    request_id = request.data_element.id
    logger.info(f"Enhancement request received for data element: {request_id}")
    
    # Get job from in-memory store
    job_data = job_store.get_job(request_id)
    
    if job_data:
        # Job exists, get status
        status = EnhancementStatus(job_data["status"])
        
        # If already completed or failed, return the result
        if status in [EnhancementStatus.COMPLETED, EnhancementStatus.FAILED]:
            return EnhancementResponse(
                request_id=request_id,
                status=status,
                enhanced_data=job_data["data"].get("result"),
                error_message=job_data["data"].get("error")
            )
        
        # Otherwise, return the current status
        return EnhancementResponse(
            request_id=request_id,
            status=status,
            enhanced_data=None,
            error_message=None
        )
    
    # Initialize job in memory
    job_store.store_job(
        job_id=request_id,
        job_type="enhancement",
        status=EnhancementStatus.PENDING.value,
        data={
            "request": request.dict(),
            "result": None,
            "error": None
        }
    )
    
    # Add the enhancement task to the background tasks
    background_tasks.add_task(
        run_enhancement_job,
        request_id=request_id,
        data_element=request.data_element,
        max_iterations=request.max_iterations
    )
    
    return EnhancementResponse(
        request_id=request_id,
        status=EnhancementStatus.PENDING,
        enhanced_data=None,
        error_message=None
    )

@router.post("/enhance/stream", response_class=StreamingResponse)
async def stream_enhance_data_element(
    request: EnhancementRequest,
):
    """
    Enhance a data element and stream the results as they become available.
    
    Args:
        request: Enhancement request with data element
        
    Returns:
        StreamingResponse with enhancement updates
    """
    async def enhancement_stream():
        workflow = get_workflow()
        try:
            # Stream the enhancement results
            async for result in workflow.stream_run(request.data_element, request.max_iterations):
                yield json.dumps(result) + "\n"
        except Exception as e:
            logger.error(f"Error in streaming enhancement: {e}")
            yield json.dumps({
                "status": "error",
                "message": str(e)
            }) + "\n"
    
    return StreamingResponse(
        enhancement_stream(),
        media_type="application/x-ndjson"
    )

@router.get("/enhance/{request_id}", response_model=EnhancementResponse)
async def get_enhancement_status(request_id: str):
    """
    Get the status of an enhancement job.
    
    Args:
        request_id: ID of the enhancement job
        
    Returns:
        EnhancementResponse with current status and results if available
    """
    # Get job from in-memory store
    job_data = job_store.get_job(request_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Enhancement job {request_id} not found")
    
    # Convert status string to enum
    status = EnhancementStatus(job_data["status"])
    
    return EnhancementResponse(
        request_id=request_id,
        status=status,
        enhanced_data=job_data["data"].get("result"),
        error_message=job_data["data"].get("error")
    )

@router.post("/enhance/batch", response_model=List[str])
async def batch_enhance_data_elements(
    requests: List[EnhancementRequest],
    background_tasks: BackgroundTasks,
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
    request_ids = []
    
    # Prepare batch job storage for better performance
    batch_jobs = []
    
    for request in requests:
        request_id = request.data_element.id
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
                "request": request.dict(),
                "result": None,
                "error": None
            }
        ))
        
        # Add the enhancement task to the background tasks
        background_tasks.add_task(
            run_enhancement_job,
            request_id=request_id,
            data_element=request.data_element,
            max_iterations=request.max_iterations
        )
    
    # Bulk store jobs for better performance
    if batch_jobs:
        job_store.bulk_store_jobs(batch_jobs)
    
    return request_ids

@router.delete("/enhance/{request_id}", response_model=Dict[str, Any])
async def delete_enhancement_job(request_id: str):
    """
    Delete an enhancement job from the system.
    
    Args:
        request_id: ID of the job to delete
        
    Returns:
        Dict with deletion message
    """
    # Get job from in-memory store
    job_data = job_store.get_job(request_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Enhancement job {request_id} not found")
    
    # Don't allow deleting running jobs
    if job_data["status"] == EnhancementStatus.IN_PROGRESS.value:
        raise HTTPException(status_code=400, detail=f"Cannot delete a job that is currently in progress")
    
    # Delete job
    job_store.delete_job(request_id)
    
    return {"message": f"Enhancement job {request_id} deleted successfully"}

async def run_enhancement_job(request_id: str, data_element: DataElement, max_iterations: int = 5):
    """
    Run the enhancement job in the background.
    
    Args:
        request_id: ID of the enhancement job
        data_element: The data element to enhance
        max_iterations: Maximum number of enhancement iterations
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