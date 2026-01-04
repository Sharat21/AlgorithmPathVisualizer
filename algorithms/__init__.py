"""
Algorithms package for pathfinding visualizations.

This package contains implementations of various search algorithms.
Each algorithm module must export:
- ALGO_NAME: Human-readable name
- ALGO_CATEGORY: "Uninformed" or "Informed"
- DEFAULT_CONFIG: Dictionary of default configuration
- find_path(grid, start, goal, config, visit_callback, yield_steps): Main function
"""

import importlib
import os
from typing import List, Dict, Any

# List of available algorithm modules
ALGORITHM_MODULES = [
    "dfs",
    "bfs",
    "ucs",
    "ids",
    "gbfs",
    "astar",
    "ida_star",
]


def discover_algorithms() -> List[Dict[str, Any]]:
    """
    Discover and load all available algorithms.
    
    Returns:
        List of dictionaries with algorithm metadata
    """
    algorithms = []
    package_dir = os.path.dirname(__file__)
    
    for module_name in ALGORITHM_MODULES:
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            
            if hasattr(module, "ALGO_NAME") and hasattr(module, "ALGO_CATEGORY"):
                algorithms.append({
                    "module": module,
                    "name": module.ALGO_NAME,
                    "category": module.ALGO_CATEGORY,
                    "default_config": getattr(module, "DEFAULT_CONFIG", {}),
                    "module_name": module_name
                })
        except (ImportError, AttributeError) as e:
            # Skip modules that can't be imported or don't have required attributes
            print(f"Warning: Could not load algorithm {module_name}: {e}")
            continue
    
    return algorithms


def get_algorithm_by_name(name: str):
    """
    Get algorithm module by name.
    
    Args:
        name: Algorithm name (ALGO_NAME)
        
    Returns:
        Algorithm module or None
    """
    algorithms = discover_algorithms()
    for algo in algorithms:
        if algo["name"] == name:
            return algo["module"]
    return None

