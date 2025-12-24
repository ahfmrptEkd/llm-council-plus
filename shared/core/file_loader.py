"""Generic file loading utilities for the shared kernel."""

import json
from pathlib import Path
from typing import Any, Dict

# Use relative imports within shared module
from .logger import setup_logger

logger = setup_logger("file_loader")


def load_json(file_path: str | Path) -> Dict[str, Any]:
    """Load a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the JSON data, or empty dict on error

    Raises:
        None - logs warnings/errors and returns empty dict
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return {}
