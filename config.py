import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Please create it from config.template.yaml"
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_environment(config: Dict[str, Any]) -> None:
    """
    Setup environment variables from config.
    
    Args:
        config: Configuration dictionary
    """
    if 'openai' in config and 'api_key' in config['openai']:
        os.environ['OPENAI_API_KEY'] = config['openai']['api_key']
