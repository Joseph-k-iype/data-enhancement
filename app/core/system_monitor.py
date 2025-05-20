"""
System monitoring utilities for the enhancement API.
"""

import os
import logging
import threading
import time
import psutil
from typing import Optional

logger = logging.getLogger(__name__)

# Global threading variables
_monitor_thread = None
_stop_monitor = threading.Event()

def get_system_stats():
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

def count_jobs_by_type():
    """
    Count jobs by type in the database.
    
    Returns:
        dict: Dictionary with job counts by type
    """
    try:
        # Import here to avoid circular imports
        from app.core.db_manager import DBManager
        db = DBManager()
        
        # Get enhancement jobs
        enhancement_jobs = db.get_jobs_by_type_and_status("enhancement")
        
        return {
            "enhancement_jobs_count": len(enhancement_jobs)
        }
    except Exception as e:
        logger.error(f"Error counting jobs: {e}")
        return {
            "enhancement_jobs_count": 0,
            "error": str(e)
        }

def record_stats_to_db():
    """Record system stats to database."""
    try:
        # Import here to avoid circular imports
        from app.core.db_manager import DBManager
        db = DBManager()
        
        # Get system stats
        stats = get_system_stats()
        
        # Get job counts
        job_counts = count_jobs_by_type()
        
        # Record stats
        db.record_system_stats(
            cpu_usage=stats["cpu_usage"],
            memory_usage=stats["memory_usage"],
            enhancement_jobs_count=job_counts["enhancement_jobs_count"]
        )
        
        logger.debug("Recorded system stats to database")
    except Exception as e:
        logger.error(f"Error recording stats to database: {e}")

def monitoring_worker(interval: int):
    """
    Worker function for the monitoring thread.
    
    Args:
        interval: How often to record stats (seconds)
    """
    logger.info(f"Starting system monitoring worker with interval {interval}s")
    
    while not _stop_monitor.is_set():
        try:
            record_stats_to_db()
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