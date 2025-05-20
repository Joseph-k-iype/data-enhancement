"""
Optimized in-memory job store to replace PostgreSQL database operations.
"""

import logging
import threading
import time
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class OptimizedInMemoryJobStore:
    """
    Optimized in-memory job store with better concurrency handling.
    """
    
    _instance = None
    _lock = threading.RLock()  # Reentrant lock for better performance
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of the job store."""
        if cls._instance is None:
            cls._instance = super(OptimizedInMemoryJobStore, cls).__new__(cls)
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
                self._cached_jobs_by_type = {}  # job_type -> list of jobs
                self._job_indexes = {}  # job_type -> {status -> set of job_ids}
                self._cache_valid = False  # Flag to indicate if cache is valid
                self._cache_lock = threading.Lock()  # Lock for cache updates
                self._executor = ThreadPoolExecutor(max_workers=4)  # Thread pool for background tasks
                self._initialized = True
                logger.info("Optimized in-memory job store initialized")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the job store with minimal locking.
        """
        # Use a local copy to minimize lock time
        with self._lock:
            job_count = len(self._jobs)
            stats_count = len(self._stats)
            
        return {
            "status": "healthy",
            "job_count": job_count,
            "stats_count": stats_count,
            "storage_type": "optimized-in-memory",
            "cache_valid": self._cache_valid
        }
    
    def store_job(self, job_id: str, job_type: str, status: str, data: Dict[str, Any]) -> bool:
        """
        Store a job in memory with optimized locking.
        """
        now = datetime.now()
        job_updated = False
        job_added = False
        
        with self._lock:
            # Check if job exists already
            if job_id in self._jobs:
                job = self._jobs[job_id]
                old_status = job["status"]
                job["status"] = status
                job["data"] = data
                job["updated_at"] = now
                job_updated = True
                
                # Update indexes if status changed
                if old_status != status and job_type in self._job_indexes:
                    if old_status in self._job_indexes[job_type]:
                        self._job_indexes[job_type][old_status].discard(job_id)
                    if status not in self._job_indexes[job_type]:
                        self._job_indexes[job_type][status] = set()
                    self._job_indexes[job_type][status].add(job_id)
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
                job_added = True
                
                # Update indexes
                if job_type not in self._job_indexes:
                    self._job_indexes[job_type] = {}
                if status not in self._job_indexes[job_type]:
                    self._job_indexes[job_type][status] = set()
                self._job_indexes[job_type][status].add(job_id)
            
            # Mark cache as invalid
            with self._cache_lock:
                self._cache_valid = False
            
        # Schedule cache update in background
        if job_added or job_updated:
            self._executor.submit(self._update_cache)
            
        return True
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job from memory with minimal locking.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                # Return a copy to avoid concurrency issues
                return job.copy()
            return None
    
    def _update_cache(self):
        """
        Update the cached job lists in the background.
        """
        if self._cache_valid:
            return
            
        try:
            # Create new caches for each job type
            with self._lock:
                job_types = set(job["job_type"] for job in self._jobs.values())
                jobs_by_type = {}
                
                for job_type in job_types:
                    jobs_of_type = [
                        job.copy() for job in self._jobs.values() 
                        if job["job_type"] == job_type
                    ]
                    
                    # Sort by updated_at (descending)
                    jobs_of_type.sort(key=lambda x: x["updated_at"], reverse=True)
                    jobs_by_type[job_type] = jobs_of_type
            
            # Update the cache
            with self._cache_lock:
                self._cached_jobs_by_type = jobs_by_type
                self._cache_valid = True
                
            logger.debug("Job cache updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating job cache: {e}")
    
    def get_jobs_by_type_and_status(self, job_type: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get jobs by type and optional status using the cached list when possible.
        """
        # Try to use indexes for faster retrieval if status is specified
        if status is not None:
            with self._lock:
                if job_type in self._job_indexes and status in self._job_indexes[job_type]:
                    # Get job IDs from index
                    job_ids = self._job_indexes[job_type][status]
                    # Get job data
                    return [self._jobs[job_id].copy() for job_id in job_ids]
        
        # Ensure cache is updated for all job types
        if not self._cache_valid:
            self._update_cache()
        
        # Try to use the cached list
        with self._cache_lock:
            if job_type in self._cached_jobs_by_type:
                jobs = self._cached_jobs_by_type[job_type]
                
                # Filter by status if needed
                if status is not None:
                    return [job.copy() for job in jobs if job["status"] == status]
                else:
                    return [job.copy() for job in jobs]
            
        # Fallback to direct filtering if cache isn't helpful
        with self._lock:
            result = []
            for job in self._jobs.values():
                if job["job_type"] == job_type:
                    if status is None or job["status"] == status:
                        result.append(job.copy())
            
            # Sort by updated_at (descending)
            result.sort(key=lambda x: x["updated_at"], reverse=True)
            return result
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from memory with cache invalidation.
        """
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job_type = job["job_type"]
                status = job["status"]
                
                # Update indexes
                if job_type in self._job_indexes and status in self._job_indexes[job_type]:
                    self._job_indexes[job_type][status].discard(job_id)
                
                # Delete the job
                del self._jobs[job_id]
                
                # Mark cache as invalid
                with self._cache_lock:
                    self._cache_valid = False
                
                # Schedule cache update in background
                self._executor.submit(self._update_cache)
                
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
            return [stat.copy() for stat in sorted_stats[:limit]]
    
    async def async_store_job(self, job_id: str, job_type: str, status: str, data: Dict[str, Any]) -> bool:
        """
        Asynchronous version of store_job for use in async contexts.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.store_job(job_id, job_type, status, data)
        )
    
    async def async_get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Asynchronous version of get_job for use in async contexts.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_job(job_id)
        )
    
    async def async_get_jobs_by_type_and_status(self, job_type: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Asynchronous version of get_jobs_by_type_and_status.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_jobs_by_type_and_status(job_type, status)
        )
    
    async def async_delete_job(self, job_id: str) -> bool:
        """
        Asynchronous version of delete_job.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.delete_job(job_id)
        )
    
    def bulk_store_jobs(self, jobs: List[Tuple[str, str, str, Dict[str, Any]]]) -> bool:
        """
        Store multiple jobs at once for better performance.
        
        Args:
            jobs: List of (job_id, job_type, status, data) tuples
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            now = datetime.now()
            
            for job_id, job_type, status, data in jobs:
                # Check if job exists already
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    old_status = job["status"]
                    job["status"] = status
                    job["data"] = data
                    job["updated_at"] = now
                    
                    # Update indexes if status changed
                    if old_status != status and job_type in self._job_indexes:
                        if old_status in self._job_indexes[job_type]:
                            self._job_indexes[job_type][old_status].discard(job_id)
                        if status not in self._job_indexes[job_type]:
                            self._job_indexes[job_type][status] = set()
                        self._job_indexes[job_type][status].add(job_id)
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
                    
                    # Update indexes
                    if job_type not in self._job_indexes:
                        self._job_indexes[job_type] = {}
                    if status not in self._job_indexes[job_type]:
                        self._job_indexes[job_type][status] = set()
                    self._job_indexes[job_type][status].add(job_id)
            
            # Mark cache as invalid
            with self._cache_lock:
                self._cache_valid = False
            
            # Schedule cache update in background
            self._executor.submit(self._update_cache)
            
            return True

# Create a global instance
optimized_job_store = OptimizedInMemoryJobStore()

# For backward compatibility, create an alias to the original class name
class InMemoryJobStore(OptimizedInMemoryJobStore):
    """Alias for OptimizedInMemoryJobStore to maintain backward compatibility."""
    pass

# Global instance for the application to use
job_store = optimized_job_store