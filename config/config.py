import json
import logging
from typing import Dict
from pathlib import Path
import os

logger = logging.getLogger(__name__)

def load_config(config_path: str = os.path.join(os.path.dirname(__file__), 'config.json')) -> Dict:
    """Load configuration file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dict containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise