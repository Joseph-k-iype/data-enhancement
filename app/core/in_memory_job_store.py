"""
In-memory job store to replace PostgreSQL database operations.
"""

import logging
import threading
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class InMemoryJobStore:
    """
    In-memory job store to replace PostgreSQL database operations.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of the job store."""
        if cls._instance is None:
            cls._instance = super(InMemoryJobStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the job store."""
        if self._initialized:
            return
            
        with self._lock:
            if not self._initialized:
                self._jobs = {}  # job_id -> job_data
                self._stats = []  # List of system stats entries
                self._initialized = True
                logger.info("In-memory job store initialized")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the job store.
        
        Returns:
            Dict: Health check information
        """
        with self._lock:
            return {
                "status": "healthy",
                "job_count": len(self._jobs),
                "stats_count": len(self._stats),
                "storage_type": "in-memory"
            }
    
    def store_job(self, job_id: str, job_type: str, status: str, data: Dict[str, Any]) -> bool:
        """
        Store a job in memory.
        
        Args:
            job_id: Unique identifier for the job
            job_type: Type of job (enhancement, etc.)
            status: Status of the job
            data: Job data as a dictionary
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            now = datetime.now()
            
            # Check if job exists already
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job["status"] = status
                job["data"] = data
                job["updated_at"] = now
            else:
                # Create new job
                self._jobs[job_id] = {
                    "id": job_id,
                    "job_type": job_type,
                    "status": status,
                    "data": data,
                    "created_at": now,
                    "updated_at": now
                }
            
            return True
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job from memory.
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            Dict: Job details or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_jobs_by_type_and_status(self, job_type: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get jobs by type and optional status.
        
        Args:
            job_type: Type of jobs to retrieve
            status: Optional status filter
            
        Returns:
            List: List of job details
        """
        with self._lock:
            result = []
            
            for job_id, job in self._jobs.items():
                if job["job_type"] == job_type:
                    if status is None or job["status"] == status:
                        result.append(job.copy())
            
            # Sort by updated_at (descending)
            result.sort(key=lambda x: x["updated_at"], reverse=True)
            
            return result
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from memory.
        
        Args:
            job_id: ID of the job to delete
            
        Returns:
            bool: True if successful, False if job not found
        """
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False
    
    def record_system_stats(self, cpu_usage: float, memory_usage: float, 
                           enhancement_jobs_count: int) -> bool:
        """
        Record system statistics in memory.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            enhancement_jobs_count: Number of enhancement jobs
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            now = datetime.now()
            
            # Add stats entry
            self._stats.append({
                "timestamp": now,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "enhancement_jobs_count": enhancement_jobs_count
            })
            
            # Keep only the latest 1000 entries to avoid memory growth
            if len(self._stats) > 1000:
                self._stats = self._stats[-1000:]
            
            return True
    
    def get_system_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent system statistics.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List: List of system stats entries
        """
        with self._lock:
            # Sort by timestamp (descending) and take the latest 'limit' entries
            sorted_stats = sorted(self._stats, key=lambda x: x["timestamp"], reverse=True)
            return sorted_stats[:limit]

# Create a global instance for the application to use
job_store = InMemoryJobStore()