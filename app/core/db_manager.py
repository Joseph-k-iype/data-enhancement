"""
PostgreSQL Database Manager with schema awareness.
Handles connection pooling, database initialization, and job tracking.
"""

import logging
import os
import time
import re
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator, Tuple

import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values, DictCursor, Json
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.environment import get_os_env

logger = logging.getLogger(__name__)

Base = declarative_base()

# Schema name - this should match what you used in setup_db.py
SCHEMA_NAME = "ai_stitching_platform"

class DBManager:
    """PostgreSQL Database Manager with schema awareness."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of the database connection pool."""
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
            # Initialize tracking variables here to avoid race conditions
            cls._instance.connection_attempts = 0
            cls._instance.successful_connections = 0
            cls._instance.failed_connections = 0
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the PostgreSQL connection."""
        if self._initialized:
            return
        
        # Ensure connection tracking attributes exist
        if not hasattr(self, 'connection_attempts'):
            self.connection_attempts = 0
        if not hasattr(self, 'successful_connections'):
            self.successful_connections = 0
        if not hasattr(self, 'failed_connections'):
            self.failed_connections = 0
        
        self.env = get_os_env()
        
        # Schema name - can be overridden by environment variable
        self.schema_name = os.environ.get("PG_SCHEMA", SCHEMA_NAME)
        
        # Get PostgreSQL connection details
        self.pg_host = self.env.get("PG_HOST", "localhost")
        self.pg_port = int(self.env.get("PG_PORT", "5432"))
        self.pg_user = self.env.get("PG_USER", "postgres")
        self.pg_password = self.env.get("PG_PASSWORD", "postgres")
        self.pg_db = self.env.get("PG_DB", "metadata_db")
        
        # Configure connection pooling
        self.min_connections = int(self.env.get("PG_MIN_CONNECTIONS", "2"))
        self.max_connections = int(self.env.get("PG_MAX_CONNECTIONS", "10"))
        
        # Initialize connection pool and engine
        self._setup_connection_pool()
        self._setup_sqlalchemy()
        
        # Initialize database schema if needed
        try:
            self._verify_schema()
        except Exception as e:
            logger.error(f"Error verifying schema: {e}")
            # Continue anyway - the schema might already exist
        
        self._initialized = True
        logger.info(f"Database manager initialized with host={self.pg_host}, port={self.pg_port}, db={self.pg_db}, schema={self.schema_name}")
    
    def _setup_connection_pool(self):
        """Set up a connection pool for PostgreSQL."""
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.pg_host,
                port=self.pg_port,
                user=self.pg_user,
                password=self.pg_password,
                database=self.pg_db
            )
            
            # Set search path for all connections in the pool
            for i in range(self.min_connections):
                conn = self.connection_pool.getconn()
                try:
                    with conn.cursor() as cursor:
                        # Set search path to include both schemas
                        cursor.execute(f"SET search_path TO {self.schema_name}, public")
                        conn.commit()
                        logger.info(f"Initialized connection with search path: {self.schema_name}, public")
                finally:
                    self.connection_pool.putconn(conn)
                    
            logger.info(f"Connection pool created with {self.min_connections}-{self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            self.connection_pool = None
    
    def _setup_sqlalchemy(self):
        """Set up SQLAlchemy engine for ORM operations."""
        try:
            db_url = f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
            self.engine = create_engine(db_url, pool_pre_ping=True)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("SQLAlchemy engine initialized")
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine: {e}")
            self.engine = None
            self.Session = None
    
    def _verify_schema(self):
        """Verify that the schema exists and create it if it doesn't."""
        try:
            # Use direct connection to avoid recursive get_connection calls
            conn = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                user=self.pg_user,
                password=self.pg_password,
                database=self.pg_db
            )
            conn.autocommit = True  # Set autocommit for schema creation
            
            try:
                with conn.cursor() as cursor:
                    # Check if schema exists
                    cursor.execute(f"SELECT 1 FROM information_schema.schemata WHERE schema_name = '{self.schema_name}'")
                    schema_exists = cursor.fetchone() is not None
                    
                    if not schema_exists:
                        logger.warning(f"Schema '{self.schema_name}' does not exist. Creating it...")
                        cursor.execute(f"CREATE SCHEMA {self.schema_name}")
                        logger.info(f"Schema '{self.schema_name}' created successfully")
                    else:
                        logger.info(f"Schema '{self.schema_name}' already exists")
                    
                    # Set search path to include our schema
                    cursor.execute(f"SET search_path TO {self.schema_name}, public")
                    
                    # Check if jobs table exists
                    cursor.execute(f"SELECT 1 FROM information_schema.tables WHERE table_schema = '{self.schema_name}' AND table_name = 'jobs'")
                    jobs_table_exists = cursor.fetchone() is not None
                    
                    if not jobs_table_exists:
                        logger.warning(f"Table '{self.schema_name}.jobs' does not exist. Creating it...")
                        try:
                            cursor.execute(f"""
                            CREATE TABLE {self.schema_name}.jobs (
                                id VARCHAR(255) PRIMARY KEY,
                                job_type VARCHAR(50) NOT NULL,
                                status VARCHAR(50) NOT NULL,
                                data JSONB DEFAULT '{{}}',
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                            """)
                            
                            # Create index on job_type and status
                            cursor.execute(f"""
                            CREATE INDEX jobs_type_status_idx 
                            ON {self.schema_name}.jobs 
                            USING btree (job_type, status)
                            """)
                            
                            logger.info(f"Table '{self.schema_name}.jobs' created successfully")
                        except Exception as e:
                            logger.error(f"Error creating jobs table: {e}")
                    
                    # Check if system_stats table exists
                    cursor.execute(f"SELECT 1 FROM information_schema.tables WHERE table_schema = '{self.schema_name}' AND table_name = 'system_stats'")
                    stats_table_exists = cursor.fetchone() is not None
                    
                    if not stats_table_exists:
                        logger.warning(f"Table '{self.schema_name}.system_stats' does not exist. Creating it...")
                        try:
                            cursor.execute(f"""
                            CREATE TABLE {self.schema_name}.system_stats (
                                id SERIAL PRIMARY KEY,
                                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                cpu_usage FLOAT NOT NULL,
                                memory_usage FLOAT NOT NULL,
                                db_size BIGINT NOT NULL,
                                active_connections INTEGER NOT NULL,
                                enhancement_jobs_count INTEGER NOT NULL
                            )
                            """)
                            
                            # Create index on timestamp
                            cursor.execute(f"""
                            CREATE INDEX system_stats_timestamp_idx 
                            ON {self.schema_name}.system_stats 
                            USING btree (timestamp)
                            """)
                            
                            logger.info(f"Table '{self.schema_name}.system_stats' created successfully")
                        except Exception as e:
                            logger.error(f"Error creating system_stats table: {e}")
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error verifying schema: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> Generator:
        """Get a connection from the pool with automatic cleanup."""
        conn = None
        
        # Safely increment connection attempts
        if hasattr(self, 'connection_attempts'):
            self.connection_attempts += 1
        
        try:
            if self.connection_pool is None:
                # Try to recreate the pool if it's None
                self._setup_connection_pool()
                if self.connection_pool is None:
                    raise Exception("Connection pool could not be created")
                
            conn = self.connection_pool.getconn()
            
            # Set search path to include our schema
            with conn.cursor() as cursor:
                cursor.execute(f"SET search_path TO {self.schema_name}, public")
            
            # Safely increment successful connections
            if hasattr(self, 'successful_connections'):
                self.successful_connections += 1
                
            yield conn
        except Exception as e:
            # Safely increment failed connections
            if hasattr(self, 'failed_connections'):
                self.failed_connections += 1
                
            logger.error(f"Error getting database connection: {e}")
            raise
        finally:
            if conn is not None:
                self.connection_pool.putconn(conn)
    
    @contextmanager
    def get_session(self) -> Generator:
        """Get a SQLAlchemy session with automatic cleanup."""
        if self.Session is None:
            raise ValueError("SQLAlchemy Session not initialized")
            
        session = self.Session()
        try:
            # Set search path
            session.execute(text(f"SET search_path TO {self.schema_name}, public"))
            yield session
        finally:
            session.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the database connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check connection
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    
                    # Get database size
                    cursor.execute(f"SELECT pg_database_size('{self.pg_db}')")
                    db_size = cursor.fetchone()[0]
                    
                    # Get schema existence
                    cursor.execute(f"SELECT 1 FROM information_schema.schemata WHERE schema_name = '{self.schema_name}'")
                    schema_exists = cursor.fetchone() is not None
                    
                    # Get connection stats
                    cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = %s", (self.pg_db,))
                    active_connections = cursor.fetchone()[0]
                    
                    # Get the connection stats, safely handling missing attributes
                    connection_stats = {
                        "attempts": getattr(self, 'connection_attempts', 0),
                        "successful": getattr(self, 'successful_connections', 0),
                        "failed": getattr(self, 'failed_connections', 0)
                    }
                    
                    return {
                        "status": "healthy",
                        "version": version,
                        "schema_exists": schema_exists,
                        "schema_name": self.schema_name,
                        "db_size_mb": round(db_size / (1024 * 1024), 2),
                        "active_connections": active_connections,
                        "connection_pool_stats": connection_stats
                    }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def store_job(self, job_id: str, job_type: str, status: str, data: Dict[str, Any]) -> bool:
        """
        Store a job in the database.
        
        Args:
            job_id: Unique identifier for the job
            job_type: Type of job (enhancement, tagging)
            status: Status of the job
            data: Job data as a dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                    INSERT INTO {self.schema_name}.jobs (id, job_type, status, data, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) 
                    DO UPDATE SET 
                        status = EXCLUDED.status,
                        data = EXCLUDED.data,
                        updated_at = CURRENT_TIMESTAMP
                    """, (job_id, job_type, status, Json(data)))
                    
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error storing job: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job from the database.
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            Dictionary with job details or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    cursor.execute(f"""
                    SELECT id, job_type, status, created_at, updated_at, data
                    FROM {self.schema_name}.jobs
                    WHERE id = %s
                    """, (job_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        return dict(row)
                    return None
        except Exception as e:
            logger.error(f"Error getting job: {e}")
            return None
    
    def get_jobs_by_type_and_status(self, job_type: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get jobs by type and optional status.
        
        Args:
            job_type: Type of jobs to retrieve
            status: Optional status filter
            
        Returns:
            List of dictionaries with job details
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    if status:
                        cursor.execute(f"""
                        SELECT id, job_type, status, created_at, updated_at, data
                        FROM {self.schema_name}.jobs
                        WHERE job_type = %s AND status = %s
                        ORDER BY updated_at DESC
                        """, (job_type, status))
                    else:
                        cursor.execute(f"""
                        SELECT id, job_type, status, created_at, updated_at, data
                        FROM {self.schema_name}.jobs
                        WHERE job_type = %s
                        ORDER BY updated_at DESC
                        """, (job_type,))
                    
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting jobs by type and status: {e}")
            return []
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from the database.
        
        Args:
            job_id: ID of the job to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                    DELETE FROM {self.schema_name}.jobs
                    WHERE id = %s
                    """, (job_id,))
                    
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting job: {e}")
            return False
    
    def record_system_stats(self, cpu_usage: float, memory_usage: float, 
                           enhancement_jobs_count: int) -> bool:
        """
        Record system statistics for monitoring.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            enhancement_jobs_count: Number of enhancement jobs
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get database size
                    cursor.execute(f"SELECT pg_database_size('{self.pg_db}')")
                    db_size = cursor.fetchone()[0]
                    
                    # Get active connections
                    cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = %s", (self.pg_db,))
                    active_connections = cursor.fetchone()[0]
                    
                    # Insert stats
                    cursor.execute(f"""
                    INSERT INTO {self.schema_name}.system_stats 
                    (cpu_usage, memory_usage, db_size, active_connections, enhancement_jobs_count)
                    VALUES (%s, %s, %s, %s, %s)
                    """, (cpu_usage, memory_usage, db_size, active_connections, enhancement_jobs_count))
                    
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error recording system stats: {e}")
            return False
    
    def get_system_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent system statistics.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of dictionaries with system statistics
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    cursor.execute(f"""
                    SELECT timestamp, cpu_usage, memory_usage, db_size, 
                           active_connections, enhancement_jobs_count
                    FROM {self.schema_name}.system_stats
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """, (limit,))
                    
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return []