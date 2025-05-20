"""
System monitoring utilities for the enhancement API.
"""

import os
import logging
import threading
import time
import psutil
from typing import Optional, Dict, Any, List

from app.core.in_memory_job_store import job_store

logger = logging.getLogger(__name__)

# Global monitoring variables
_monitor_thread = None
_stop_monitor = threading.Event()

def get_system_stats() -> Dict[str, Any]:
    """
    Get current system resource usage stats.
    
    Returns:
        dict: Dictionary with CPU usage, memory usage, and disk usage
    """
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "disk_usage": disk_percent
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "error": str(e)
        }

def count_jobs() -> Dict[str, int]:
    """
    Count jobs by type in the in-memory job store.
    
    Returns:
        dict: Dictionary with job counts by type
    """
    try:
        # Get jobs from in-memory store
        enhancement_jobs = job_store.get_jobs_by_type_and_status("enhancement")
        active_jobs = [job for job in enhancement_jobs 
                     if job["status"] in ["pending", "in_progress"]]
        
        return {
            "total_jobs": len(enhancement_jobs),
            "active_jobs": len(active_jobs),
            "enhancement_jobs_count": len(enhancement_jobs)
        }
    except Exception as e:
        logger.error(f"Error counting jobs: {e}")
        return {
            "total_jobs": 0,
            "active_jobs": 0, 
            "enhancement_jobs_count": 0,
            "error": str(e)
        }

def record_stats() -> bool:
    """Record system stats to in-memory job store."""
    try:
        # Get system stats
        stats = get_system_stats()
        
        # Get job counts
        job_counts = count_jobs()
        
        # Record stats to in-memory store
        job_store.record_system_stats(
            cpu_usage=stats["cpu_usage"],
            memory_usage=stats["memory_usage"],
            enhancement_jobs_count=job_counts["enhancement_jobs_count"]
        )
        
        logger.debug("Recorded system stats successfully")
        return True
    except Exception as e:
        logger.error(f"Error recording stats: {e}")
        return False

def monitoring_worker(interval: int):
    """
    Worker function for the monitoring thread.
    
    Args:
        interval: How often to record stats (seconds)
    """
    logger.info(f"Starting system monitoring worker with interval {interval}s")
    
    while not _stop_monitor.is_set():
        try:
            record_stats()
        except Exception as e:
            logger.error(f"Error in monitoring worker: {e}")
        
        # Sleep until next check
        _stop_monitor.wait(interval)
    
    logger.info("System monitoring worker stopped")

def start_monitoring(interval: int = 300):
    """
    Start system monitoring in a background thread.
    
    Args:
        interval: How often to record stats (seconds)
    """
    global _monitor_thread, _stop_monitor
    
    if _monitor_thread is not None and _monitor_thread.is_alive():
        logger.info("Monitoring already running")
        return
    
    _stop_monitor.clear()
    _monitor_thread = threading.Thread(
        target=monitoring_worker,
        args=(interval,),
        daemon=True
    )
    _monitor_thread.start()
    logger.info(f"Started system monitoring thread with interval {interval}s")

def stop_monitoring():
    """Stop system monitoring."""
    global _monitor_thread, _stop_monitor
    
    if _monitor_thread is not None and _monitor_thread.is_alive():
        _stop_monitor.set()
        _monitor_thread.join(timeout=10)
        logger.info("Stopped system monitoring thread")

def get_recent_stats(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent system statistics from in-memory store.
    
    Args:
        limit: Maximum number of stats to return
        
    Returns:
        List of stats entries
    """
    return job_store.get_system_stats(limit)