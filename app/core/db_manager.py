"""
Stub implementation of DBManager to maintain compatibility.

This module provides a stub implementation of the original PostgreSQL-based 
DBManager to maintain compatibility with any code that might still reference it.
All operations now redirect to the in-memory job store.
"""

import logging
from typing import List, Dict, Any, Optional
from app.core.in_memory_job_store import job_store

logger = logging.getLogger(__name__)

class DBManager:
    """
    Stub implementation of the PostgreSQL database manager.
    All operations now use the in-memory job store.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the database manager stub."""
        if self._initialized:
            return
            
        self._initialized = True
        logger.info("DBManager stub initialized - using in-memory storage")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the database (stub implementation).
        
        Returns:
            Dict: Health check information from in-memory store
        """
        return job_store.health_check()
    
    def store_job(self, job_id: str, job_type: str, status: str, data: Dict[str, Any]) -> bool:
        """
        Store a job (redirects to in-memory store).
        
        Args:
            job_id: Unique identifier for the job
            job_type: Type of job (enhancement, etc.)
            status: Status of the job
            data: Job data as a dictionary
            
        Returns:
            bool: True if successful
        """
        return job_store.store_job(job_id, job_type, status, data)
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job (redirects to in-memory store).
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            Dict: Job details or None if not found
        """
        return job_store.get_job(job_id)
    
    def get_jobs_by_type_and_status(self, job_type: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get jobs by type and optional status (redirects to in-memory store).
        
        Args:
            job_type: Type of jobs to retrieve
            status: Optional status filter
            
        Returns:
            List: List of job details
        """
        return job_store.get_jobs_by_type_and_status(job_type, status)
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job (redirects to in-memory store).
        
        Args:
            job_id: ID of the job to delete
            
        Returns:
            bool: True if successful, False if job not found
        """
        return job_store.delete_job(job_id)
    
    def record_system_stats(self, cpu_usage: float, memory_usage: float, enhancement_jobs_count: int) -> bool:
        """
        Record system statistics (redirects to in-memory store).
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            enhancement_jobs_count: Number of enhancement jobs
            
        Returns:
            bool: True if successful
        """
        return job_store.record_system_stats(cpu_usage, memory_usage, enhancement_jobs_count)
    
    def get_system_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent system statistics (redirects to in-memory store).
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List: List of system stats entries
        """
        return job_store.get_system_stats(limit)