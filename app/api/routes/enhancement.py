"""
Enhancement API - Routes for data element enhancement.

This module provides API endpoints for enhancing data elements to meet ISO/IEC 11179
standards, with support for asynchronous processing and job tracking.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
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
from app.core.db_manager import DBManager
from app.agents.workflow import DataEnhancementWorkflow
from app.config.settings import get_llm

router = APIRouter(prefix="/api/v1", tags=["data-enhancement"])

# In-memory cache for current jobs for fast access
# The persistent storage is in PostgreSQL
enhancement_jobs: Dict[str, Dict[str, Any]] = {}

logger = logging.getLogger(__name__)

def get_workflow() -> DataEnhancementWorkflow:
    """
    Get the data enhancement workflow.
    
    Returns:
        DataEnhancementWorkflow: The enhancement workflow
    """
    llm = get_llm()
    return DataEnhancementWorkflow(llm)

def get_db():
    """
    Get the database manager.
    
    Returns:
        DBManager: The database manager
    """
    return DBManager()

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
    db: DBManager = Depends(get_db)
):
    """
    Enhance a data element to meet ISO/IEC 11179 standards.
    This is an asynchronous operation that will run in the background.
    
    Args:
        request: Enhancement request with data element
        background_tasks: FastAPI background tasks
        db: Database manager
        
    Returns:
        EnhancementResponse with request ID and status
    """
    # Use the provided ID as the request ID for tracking
    request_id = request.data_element.id
    logger.info(f"Enhancement request received for data element: {request_id}")
    
    # Check if this ID is already being processed in memory
    if request_id in enhancement_jobs:
        logger.info(f"Enhancement job already exists in memory for ID: {request_id}")
        job = enhancement_jobs[request_id]
        
        # If already completed or failed, return the result
        if job["status"] in [EnhancementStatus.COMPLETED, EnhancementStatus.FAILED]:
            return EnhancementResponse(
                request_id=request_id,
                status=job["status"],
                enhanced_data=job.get("result"),  # Use get() to avoid KeyError
                error_message=job.get("error")
            )
        
        # Otherwise, return the current status
        return EnhancementResponse(
            request_id=request_id,
            status=job["status"],
            enhanced_data=None,
            error_message=None
        )
    
    # Check if job exists in the database
    db_job = db.get_job(request_id)
    if db_job is not None:
        # Job exists in database
        status = EnhancementStatus(db_job["status"])
        
        # Load job data from database to memory for faster access
        enhancement_jobs[request_id] = {
            "status": status,
            "request": db_job["data"].get("request", {}),
            "result": db_job["data"].get("result"),
            "error": db_job["data"].get("error")
        }
        
        # If already completed or failed, return the result
        if status in [EnhancementStatus.COMPLETED, EnhancementStatus.FAILED]:
            return EnhancementResponse(
                request_id=request_id,
                status=status,
                enhanced_data=enhancement_jobs[request_id].get("result"),
                error_message=enhancement_jobs[request_id].get("error")
            )
        
        # Otherwise, return the current status
        return EnhancementResponse(
            request_id=request_id,
            status=status,
            enhanced_data=None,
            error_message=None
        )
    
    # Initialize job status in memory - important to set all fields
    enhancement_jobs[request_id] = {
        "status": EnhancementStatus.PENDING,
        "request": request.dict(),
        "result": None,
        "error": None
    }
    
    # Store job in database
    db.store_job(
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

@router.get("/enhance/{request_id}", response_model=EnhancementResponse)
async def get_enhancement_status(request_id: str, db: DBManager = Depends(get_db)):
    """
    Get the status of an enhancement job.
    
    Args:
        request_id: ID of the enhancement job
        db: Database manager
        
    Returns:
        EnhancementResponse with current status and results if available
    """
    # First check in-memory cache for faster access
    if request_id in enhancement_jobs:
        job = enhancement_jobs[request_id]
        
        logger.debug(f"Enhancement job found in memory: {request_id}, status: {job['status']}")
        logger.debug(f"Job data: {job}")
        
        return EnhancementResponse(
            request_id=request_id,
            status=job["status"],
            enhanced_data=job.get("result"),
            error_message=job.get("error")
        )
    
    # If not in memory, check the database
    db_job = db.get_job(request_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail=f"Enhancement job {request_id} not found")
    
    # Convert status string to enum
    status = EnhancementStatus(db_job["status"])
    
    # Load into memory cache for future requests
    enhancement_jobs[request_id] = {
        "status": status,
        "request": db_job["data"].get("request", {}),
        "result": db_job["data"].get("result"),
        "error": db_job["data"].get("error")
    }
    
    logger.debug(f"Enhancement job loaded from database: {request_id}, status: {status}")
    
    return EnhancementResponse(
        request_id=request_id,
        status=status,
        enhanced_data=enhancement_jobs[request_id].get("result"),
        error_message=enhancement_jobs[request_id].get("error")
    )

@router.post("/enhance/batch", response_model=List[str])
async def batch_enhance_data_elements(
    requests: List[EnhancementRequest],
    background_tasks: BackgroundTasks,
    db: DBManager = Depends(get_db)
):
    """
    Enhance multiple data elements in batch mode.
    Returns a list of request IDs that can be used to check status.
    
    Args:
        requests: List of enhancement requests
        background_tasks: FastAPI background tasks
        db: Database manager
        
    Returns:
        List of request IDs
    """
    request_ids = []
    
    for request in requests:
        request_id = request.data_element.id
        request_ids.append(request_id)
        
        # Skip if already processing in memory
        if request_id in enhancement_jobs:
            if enhancement_jobs[request_id]["status"] not in [EnhancementStatus.COMPLETED, EnhancementStatus.FAILED]:
                continue
        
        # Check if job exists in database
        db_job = db.get_job(request_id)
        if db_job is not None:
            status = EnhancementStatus(db_job["status"])
            
            # Skip if already completed or failed
            if status in [EnhancementStatus.COMPLETED, EnhancementStatus.FAILED]:
                # Load into memory cache
                enhancement_jobs[request_id] = {
                    "status": status,
                    "request": db_job["data"].get("request", {}),
                    "result": db_job["data"].get("result"),
                    "error": db_job["data"].get("error")
                }
                continue
        
        # Initialize job status with all required fields
        enhancement_jobs[request_id] = {
            "status": EnhancementStatus.PENDING,
            "request": request.dict(),
            "result": None,
            "error": None
        }
        
        # Store job in database
        db.store_job(
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
    
    return request_ids

@router.delete("/enhance/{request_id}", response_model=Dict[str, Any])
async def delete_enhancement_job(request_id: str, db: DBManager = Depends(get_db)):
    """
    Delete an enhancement job from the system.
    
    Args:
        request_id: ID of the job to delete
        db: Database manager
        
    Returns:
        Dict with deletion message
    """
    # Check if job exists
    if request_id not in enhancement_jobs and db.get_job(request_id) is None:
        raise HTTPException(status_code=404, detail=f"Enhancement job {request_id} not found")
    
    # Get status from memory or database
    status = None
    if request_id in enhancement_jobs:
        status = enhancement_jobs[request_id]["status"]
    else:
        db_job = db.get_job(request_id)
        if db_job:
            status = EnhancementStatus(db_job["status"])
    
    # Don't allow deleting running jobs
    if status == EnhancementStatus.IN_PROGRESS:
        raise HTTPException(status_code=400, detail=f"Cannot delete a job that is currently in progress")
    
    # Delete from memory
    if request_id in enhancement_jobs:
        del enhancement_jobs[request_id]
    
    # Delete from database
    db.delete_job(request_id)
    
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
    db = DBManager()
    
    try:
        # Update job status to in progress
        enhancement_jobs[request_id]["status"] = EnhancementStatus.IN_PROGRESS
        
        # Update database status
        db.store_job(
            job_id=request_id,
            job_type="enhancement",
            status=EnhancementStatus.IN_PROGRESS.value,
            data=enhancement_jobs[request_id]
        )
        
        # Run the workflow
        result = await workflow.run(data_element, max_iterations)
        
        # Update job status to completed in memory
        enhancement_jobs[request_id]["status"] = EnhancementStatus.COMPLETED
        enhancement_jobs[request_id]["result"] = result
        enhancement_jobs[request_id]["error"] = None
        
        logger.info(f"Job result: {result}")
        
        # Update database with result - important to convert result to dict
        db.store_job(
            job_id=request_id,
            job_type="enhancement",
            status=EnhancementStatus.COMPLETED.value,
            data={
                "request": enhancement_jobs[request_id]["request"],
                "result": result.dict(),
                "error": None
            }
        )
        
        logger.info(f"Enhancement job completed for {request_id}")
        
    except Exception as e:
        # Update job status to failed in memory
        logger.error(f"Enhancement job failed for {request_id}: {str(e)}")
        
        if request_id in enhancement_jobs:
            enhancement_jobs[request_id]["status"] = EnhancementStatus.FAILED
            enhancement_jobs[request_id]["error"] = str(e)
            enhancement_jobs[request_id]["result"] = None
        
            # Update database with error
            db.store_job(
                job_id=request_id,
                job_type="enhancement",
                status=EnhancementStatus.FAILED.value,
                data={
                    "request": enhancement_jobs[request_id]["request"],
                    "result": None,
                    "error": str(e)
                }
            )
        else:
            logger.error(f"Job {request_id} not found in memory after failure")