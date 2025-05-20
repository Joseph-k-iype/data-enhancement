import logging
import json
import os
import uuid
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def generate_unique_id() -> str:
    """Generate a unique ID for tracking requests."""
    return str(uuid.uuid4())

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def load_json_file(file_path: str, default: Optional[Any] = None) -> Any:
    """Load a JSON file, returning a default value if the file doesn't exist."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return default

def save_json_file(data: Any, file_path: str) -> bool:
    """Save data to a JSON file."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory_exists(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

def format_name_for_business(technical_name: str) -> str:
    """
    Convert a technical name (snake_case, camelCase, etc.) to a business-friendly 
    lowercase name with spaces.
    """
    if not technical_name:
        return ""
    
    # First, handle camelCase by inserting spaces before capital letters
    spaced_name = ""
    for char in technical_name:
        if char.isupper():
            spaced_name += f" {char.lower()}"
        else:
            spaced_name += char
    
    # Replace underscores, hyphens with spaces
    spaced_name = spaced_name.replace('_', ' ').replace('-', ' ')
    
    # Remove any double spaces
    while '  ' in spaced_name:
        spaced_name = spaced_name.replace('  ', ' ')
    
    # Ensure all lowercase
    spaced_name = spaced_name.lower().strip()
    
    return spaced_name