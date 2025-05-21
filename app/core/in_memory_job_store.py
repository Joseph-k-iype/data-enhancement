"""
Optimized in-memory job store for improved performance.
Replace app/core/in_memory_job_store.py with this implementation.
"""

import logging
import threading
import time
import asyncio
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logger = logging.getLogger(__name__)

class OptimizedInMemoryJobStore:
    """
    High-performance in-memory job store with advanced indexing and caching.
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
        """Initialize the job store with performance optimizations."""
        if self._initialized:
            return
            
        with self._lock:
            if not self._initialized:
                # Main storage
                self._jobs = {}  # job_id -> job_data
                
                # Advanced indexing
                self._jobs_by_type = {}  # job_type -> Dict[job_id, job_data] 
                self._jobs_by_type_and_status = {}  # job_type -> Dict[status, Set[job_id]]
                
                # Stats storage
                self._stats = []  # List of system stats entries
                self._stats_by_day = {}  # date -> List[stats]
                
                # Caching
                self._cache_valid = True
                self._cache_locks = {
                    "jobs_by_type": threading.Lock(),
                    "jobs_by_status": threading.Lock(),
                    "stats": threading.Lock()
                }
                
                # Background processing
                self._executor = ThreadPoolExecutor(max_workers=4)  # Thread pool for background tasks
                self._cleanup_thread = None
                self._stop_cleanup = threading.Event()
                
                # Cache expiry times
                self._cache_ttl = {
                    "jobs_by_type": 5,  # seconds
                    "jobs_by_status": 5,
                    "stats": 30
                }
                
                # Last cache update times
                self._last_cache_update = {
                    "jobs_by_type": 0,
                    "jobs_by_status": 0,
                    "stats": 0
                }
                
                # Start cleanup thread
                self._start_cleanup_thread()
                
                self._initialized = True
                logger.info("Optimized in-memory job store initialized with advanced indexing")
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup and maintenance."""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return
            
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True
        )
        self._cleanup_thread.start()
        logger.info("Job store cleanup thread started")
    
    def _cleanup_worker(self):
        """Worker function for background cleanup and maintenance."""
        cleanup_interval = 300  # 5 minutes
        cache_validate_interval = 60  # 1 minute
        
        last_cleanup = 0
        last_cache_validate = 0
        
        while not self._stop_cleanup.wait(10):
            now = time.time()
            
            # Periodic cleanup of old stats
            if now - last_cleanup >= cleanup_interval:
                try:
                    self._cleanup_old_stats()
                    last_cleanup = now
                except Exception as e:
                    logger.error(f"Error in stats cleanup: {e}")
            
            # Periodically validate cache consistency
            if now - last_cache_validate >= cache_validate_interval:
                try:
                    self._validate_cache_consistency()
                    last_cache_validate = now
                except Exception as e:
                    logger.error(f"Error validating cache consistency: {e}")
        
        logger.info("Job store cleanup thread stopped")
    
    def _cleanup_old_stats(self):
        """Clean up old system stats to prevent memory growth."""
        with self._cache_locks["stats"]:
            # Keep only the latest 1000 entries
            if len(self._stats) > 1000:
                logger.info(f"Cleaning up old stats (current count: {len(self._stats)})")
                self._stats = self._stats[-1000:]
                
                # Rebuild stats_by_day index
                self._stats_by_day = {}
                for stat in self._stats:
                    date_str = stat["timestamp"].strftime("%Y-%m-%d")
                    if date_str not in self._stats_by_day:
                        self._stats_by_day[date_str] = []
                    self._stats_by_day[date_str].append(stat)
                
                logger.info(f"Stats cleaned up, new count: {len(self._stats)}")
    
    def _validate_cache_consistency(self):
        """Validate and repair cache consistency."""
        with self._lock:
            # Check jobs_by_type cache
            rebuild_jobs_by_type = False
            for job_type, jobs in self._jobs_by_type.items():
                job_ids = set(jobs.keys())
                expected_ids = {job_id for job_id, job in self._jobs.items() 
                              if job["job_type"] == job_type}
                
                if job_ids != expected_ids:
                    rebuild_jobs_by_type = True
                    logger.warning(f"Cache inconsistency detected in jobs_by_type for {job_type}")
                    break
            
            # Check jobs_by_type_and_status cache
            rebuild_status_index = False
            for job_type, status_dict in self._jobs_by_type_and_status.items():
                for status, job_ids in status_dict.items():
                    expected_ids = {job_id for job_id, job in self._jobs.items() 
                                  if job["job_type"] == job_type and job["status"] == status}
                    
                    if set(job_ids) != expected_ids:
                        rebuild_status_index = True
                        logger.warning(f"Cache inconsistency detected in status index for {job_type}/{status}")
                        break
                if rebuild_status_index:
                    break
            
            # Rebuild caches if needed
            if rebuild_jobs_by_type:
                self._rebuild_jobs_by_type_cache()
            
            if rebuild_status_index:
                self._rebuild_status_index()
            
            # Mark cache as valid
            self._cache_valid = True
    
    def _rebuild_jobs_by_type_cache(self):
        """Rebuild the jobs_by_type cache from scratch."""
        # This method is called under lock
        logger.info("Rebuilding jobs_by_type cache")
        
        new_jobs_by_type = {}
        
        for job_id, job in self._jobs.items():
            job_type = job["job_type"]
            
            if job_type not in new_jobs_by_type:
                new_jobs_by_type[job_type] = {}
            
            new_jobs_by_type[job_type][job_id] = job
        
        self._jobs_by_type = new_jobs_by_type
        self._last_cache_update["jobs_by_type"] = time.time()
        logger.info("jobs_by_type cache rebuilt")
    
    def _rebuild_status_index(self):
        """Rebuild the status index from scratch."""
        # This method is called under lock
        logger.info("Rebuilding status index")
        
        new_index = {}
        
        for job_id, job in self._jobs.items():
            job_type = job["job_type"]
            status = job["status"]
            
            if job_type not in new_index:
                new_index[job_type] = {}
            
            if status not in new_index[job_type]:
                new_index[job_type][status] = set()
            
            new_index[job_type][status].add(job_id)
        
        self._jobs_by_type_and_status = new_index
        self._last_cache_update["jobs_by_status"] = time.time()
        logger.info("Status index rebuilt")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the job store with minimal locking.
        
        Returns:
            Dict: Health status information
        """
        # Use local copies to minimize lock time
        with self._lock:
            job_count = len(self._jobs)
            job_types = len(self._jobs_by_type)
            
        with self._cache_locks["stats"]:
            stats_count = len(self._stats)
        
        # Calculate memory usage estimate
        memory_usage = self._estimate_memory_usage()
        
        return {
            "status": "healthy",
            "job_count": job_count,
            "job_types": job_types,
            "stats_count": stats_count,
            "storage_type": "optimized-in-memory",
            "cache_valid": self._cache_valid,
            "memory_usage_bytes": memory_usage,
            "memory_usage_mb": memory_usage / (1024 * 1024),
            "version": "2.1.0"
        }
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of the job store in bytes.
        
        Returns:
            int: Estimated memory usage in bytes
        """
        # Sample a few jobs to estimate average size
        job_size = 0
        sample_size = min(10, len(self._jobs))
        
        if sample_size > 0:
            with self._lock:
                sampled_jobs = list(self._jobs.values())[:sample_size]
                
                for job in sampled_jobs:
                    try:
                        # Use pickle to estimate size
                        job_size += len(pickle.dumps(job))
                    except:
                        # Fallback estimation
                        job_size += 2048  # Assume 2KB per job
                
                # Extrapolate to all jobs
                avg_job_size = job_size / sample_size
                job_memory = avg_job_size * len(self._jobs)
        else:
            job_memory = 0
        
        # Estimate stats memory
        with self._cache_locks["stats"]:
            stats_size = 0
            sample_size = min(10, len(self._stats))
            
            if sample_size > 0:
                sampled_stats = self._stats[:sample_size]
                
                for stat in sampled_stats:
                    try:
                        stats_size += len(pickle.dumps(stat))
                    except:
                        stats_size += 512  # Assume 512 bytes per stat
                
                # Extrapolate to all stats
                avg_stat_size = stats_size / sample_size
                stats_memory = avg_stat_size * len(self._stats)
            else:
                stats_memory = 0
        
        # Account for indexes (rough estimate)
        index_memory = len(self._jobs) * 100  # Estimate 100 bytes per job in indexes
        
        return int(job_memory + stats_memory + index_memory)
    
    def store_job(self, job_id: str, job_type: str, status: str, data: Dict[str, Any]) -> bool:
        """
        Store a job with optimized indexing.
        
        Args:
            job_id: Unique identifier for the job
            job_type: Type of job
            status: Status of the job
            data: Job data
            
        Returns:
            bool: True if successful
        """
        now = datetime.now()
        job_updated = False
        job_added = False
        old_status = None
        
        with self._lock:
            # Check if job exists already
            if job_id in self._jobs:
                job = self._jobs[job_id]
                old_status = job["status"]
                job["status"] = status
                job["data"] = data
                job["updated_at"] = now
                job_updated = True
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
            
            # Update type index
            if job_type in self._jobs_by_type:
                if job_added:
                    self._jobs_by_type[job_type][job_id] = self._jobs[job_id]
                elif job_updated:
                    self._jobs_by_type[job_type][job_id] = self._jobs[job_id]
            else:
                self._jobs_by_type[job_type] = {job_id: self._jobs[job_id]}
            
            # Update status index
            if job_type not in self._jobs_by_type_and_status:
                self._jobs_by_type_and_status[job_type] = {}
            
            # Remove from old status set if status changed
            if old_status and old_status != status:
                if old_status in self._jobs_by_type_and_status[job_type]:
                    self._jobs_by_type_and_status[job_type][old_status].discard(job_id)
            
            # Add to new status set
            if status not in self._jobs_by_type_and_status[job_type]:
                self._jobs_by_type_and_status[job_type][status] = set()
            
            self._jobs_by_type_and_status[job_type][status].add(job_id)
            
        # Schedule cache update in background if necessary
        if job_added or (job_updated and old_status != status):
            self._last_cache_update["jobs_by_type"] = time.time()
            self._last_cache_update["jobs_by_status"] = time.time()
            
        return True
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job directly with minimal locking.
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            Dict: Job data or None if not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                # Return a deep copy to avoid concurrency issues
                return pickle.loads(pickle.dumps(job))
            return None
    
    def get_jobs_by_type_and_status(self, job_type: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get jobs by type and optional status using optimized indexes.
        
        Args:
            job_type: Type of jobs to retrieve
            status: Optional status filter
            
        Returns:
            List: List of jobs matching criteria
        """
        now = time.time()
        
        # Check if index needs updating
        with self._lock:
            jobs_by_type_stale = (now - self._last_cache_update["jobs_by_type"] > self._cache_ttl["jobs_by_type"])
            jobs_by_status_stale = (now - self._last_cache_update["jobs_by_status"] > self._cache_ttl["jobs_by_status"])
            
            if jobs_by_type_stale:
                self._rebuild_jobs_by_type_cache()
            
            if jobs_by_status_stale:
                self._rebuild_status_index()
            
            # Fast path using status index for filtering by status
            if status is not None:
                if job_type in self._jobs_by_type_and_status and status in self._jobs_by_type_and_status[job_type]:
                    job_ids = self._jobs_by_type_and_status[job_type][status]
                    result = [pickle.loads(pickle.dumps(self._jobs[job_id])) for job_id in job_ids]
                    
                    # Sort by updated_at (descending)
                    result.sort(key=lambda x: x["updated_at"], reverse=True)
                    return result
                else:
                    return []  # No jobs found with this type and status
            
            # Retrieve all jobs of the given type
            if job_type in self._jobs_by_type:
                jobs = list(self._jobs_by_type[job_type].values())
                
                # Deep copy to avoid concurrency issues
                result = [pickle.loads(pickle.dumps(job)) for job in jobs]
                
                # Sort by updated_at (descending)
                result.sort(key=lambda x: x["updated_at"], reverse=True)
                return result
            
            return []  # No jobs found with this type
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job with index updates.
        
        Args:
            job_id: ID of the job to delete
            
        Returns:
            bool: True if successful, False if job not found
        """
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job_type = job["job_type"]
                status = job["status"]
                
                # Remove from main storage
                del self._jobs[job_id]
                
                # Remove from type index
                if job_type in self._jobs_by_type and job_id in self._jobs_by_type[job_type]:
                    del self._jobs_by_type[job_type][job_id]
                
                # Remove from status index
                if (job_type in self._jobs_by_type_and_status and 
                    status in self._jobs_by_type_and_status[job_type]):
                    self._jobs_by_type_and_status[job_type][status].discard(job_id)
                
                # Update cache timestamps
                self._last_cache_update["jobs_by_type"] = time.time()
                self._last_cache_update["jobs_by_status"] = time.time()
                
                return True
            return False
    
    def record_system_stats(self, cpu_usage: float, memory_usage: float, 
                           enhancement_jobs_count: int) -> bool:
        """
        Record system statistics with date-based indexing.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            enhancement_jobs_count: Number of enhancement jobs
            
        Returns:
            bool: True if successful
        """
        with self._cache_locks["stats"]:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            
            # Create stat entry
            stat_entry = {
                "timestamp": now,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "enhancement_jobs_count": enhancement_jobs_count
            }
            
            # Add to main stats list
            self._stats.append(stat_entry)
            
            # Add to date-based index
            if date_str not in self._stats_by_day:
                self._stats_by_day[date_str] = []
            self._stats_by_day[date_str].append(stat_entry)
            
            # Keep only latest entries to avoid memory growth
            if len(self._stats) > 1000:
                self._stats = self._stats[-1000:]
                # Will rebuild date index during next cleanup
            
            return True
    
    @lru_cache(maxsize=16)
    def get_system_stats(self, limit: int = 100, days: int = 1) -> List[Dict[str, Any]]:
        """
        Get recent system statistics with improved caching.
        
        Args:
            limit: Maximum number of stats to return
            days: Number of days to include
            
        Returns:
            List: List of stats entries
        """
        with self._cache_locks["stats"]:
            # Get all stats for the requested number of days
            result = []
            
            if days == 1:
                # Fast path for most common case (today only)
                today = datetime.now().strftime("%Y-%m-%d")
                if today in self._stats_by_day:
                    result = self._stats_by_day[today]
            else:
                # Get stats for multiple days
                for i in range(days):
                    date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                    if date in self._stats_by_day:
                        result.extend(self._stats_by_day[date])
            
            # Sort by timestamp (descending) and take latest 'limit'
            result.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Deep copy to avoid concurrency issues
            return pickle.loads(pickle.dumps(result[:limit]))
    
    async def async_store_job(self, job_id: str, job_type: str, status: str, data: Dict[str, Any]) -> bool:
        """
        Asynchronous version of store_job for use in async contexts.
        
        Args:
            job_id: ID of the job
            job_type: Type of job
            status: Status of the job
            data: Job data
            
        Returns:
            bool: True if successful
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            lambda: self.store_job(job_id, job_type, status, data)
        )
    
    async def async_get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Asynchronous version of get_job for use in async contexts.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dict: Job data or None if not found
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.get_job(job_id)
        )
    
    async def async_get_jobs_by_type_and_status(self, job_type: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Asynchronous version of get_jobs_by_type_and_status.
        
        Args:
            job_type: Type of jobs
            status: Optional status filter
            
        Returns:
            List: List of matching jobs
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.get_jobs_by_type_and_status(job_type, status)
        )
    
    async def async_delete_job(self, job_id: str) -> bool:
        """
        Asynchronous version of delete_job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            bool: True if successful
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
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
            
            # Track unique job types and statuses for index updates
            affected_types = set()
            
            for job_id, job_type, status, data in jobs:
                affected_types.add(job_type)
                
                # Check if job exists already
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    old_status = job["status"]
                    job["status"] = status
                    job["data"] = data
                    job["updated_at"] = now
                    
                    # Update status index if status changed
                    if old_status != status:
                        if job_type in self._jobs_by_type_and_status and old_status in self._jobs_by_type_and_status[job_type]:
                            self._jobs_by_type_and_status[job_type][old_status].discard(job_id)
                        
                        if job_type not in self._jobs_by_type_and_status:
                            self._jobs_by_type_and_status[job_type] = {}
                        
                        if status not in self._jobs_by_type_and_status[job_type]:
                            self._jobs_by_type_and_status[job_type][status] = set()
                        
                        self._jobs_by_type_and_status[job_type][status].add(job_id)
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
                    
                    # Update type index
                    if job_type in self._jobs_by_type:
                        self._jobs_by_type[job_type][job_id] = self._jobs[job_id]
                    else:
                        self._jobs_by_type[job_type] = {job_id: self._jobs[job_id]}
                    
                    # Update status index
                    if job_type not in self._jobs_by_type_and_status:
                        self._jobs_by_type_and_status[job_type] = {}
                    
                    if status not in self._jobs_by_type_and_status[job_type]:
                        self._jobs_by_type_and_status[job_type][status] = set()
                    
                    self._jobs_by_type_and_status[job_type][status].add(job_id)
            
            # Mark caches as fresh
            current_time = time.time()
            self._last_cache_update["jobs_by_type"] = current_time
            self._last_cache_update["jobs_by_status"] = current_time
            
            return True
    
    def get_job_count(self, job_type: Optional[str] = None, status: Optional[str] = None) -> int:
        """
        Get count of jobs matching criteria.
        
        Args:
            job_type: Optional job type filter
            status: Optional status filter
            
        Returns:
            int: Count of matching jobs
        """
        with self._lock:
            if job_type is None:
                # Count all jobs
                return len(self._jobs)
            
            if status is None:
                # Count jobs of specific type
                if job_type in self._jobs_by_type:
                    return len(self._jobs_by_type[job_type])
                return 0
            
            # Count jobs of specific type and status
            if (job_type in self._jobs_by_type_and_status and 
                status in self._jobs_by_type_and_status[job_type]):
                return len(self._jobs_by_type_and_status[job_type][status])
            
            return 0
    
    def clear_old_jobs(self, job_type: str, older_than_days: int = 7) -> int:
        """
        Clear jobs older than specified days.
        
        Args:
            job_type: Type of jobs to clear
            older_than_days: Age threshold in days
            
        Returns:
            int: Number of jobs cleared
        """
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        jobs_cleared = 0
        
        with self._lock:
            if job_type not in self._jobs_by_type:
                return 0
            
            job_ids_to_remove = []
            
            # Find jobs to remove
            for job_id, job in self._jobs_by_type[job_type].items():
                updated_at = job.get("updated_at")
                if updated_at and updated_at < cutoff_time:
                    job_ids_to_remove.append(job_id)
            
            # Remove them
            for job_id in job_ids_to_remove:
                status = self._jobs[job_id]["status"]
                
                # Remove from main storage
                del self._jobs[job_id]
                
                # Remove from type index
                del self._jobs_by_type[job_type][job_id]
                
                # Remove from status index
                if status in self._jobs_by_type_and_status[job_type]:
                    self._jobs_by_type_and_status[job_type][status].discard(job_id)
                
                jobs_cleared += 1
            
            # Mark caches as fresh
            if jobs_cleared > 0:
                self._last_cache_update["jobs_by_type"] = time.time()
                self._last_cache_update["jobs_by_status"] = time.time()
            
            return jobs_cleared

    def stop(self):
        """Stop background threads and cleanup."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
        
        self._executor.shutdown(wait=False)
        logger.info("Job store stopped")

# Create a global instance
optimized_job_store = OptimizedInMemoryJobStore()

# For backward compatibility, create an alias to the original class name
class InMemoryJobStore(OptimizedInMemoryJobStore):
    """Alias for OptimizedInMemoryJobStore to maintain backward compatibility."""
    pass

# Global instance for the application to use
job_store = optimized_job_store