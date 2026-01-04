"""
Breadth-First Search (BFS) algorithm implementation.
"""

import time
from typing import Tuple, List, Optional, Dict, Callable, Iterator
from collections import deque

ALGO_NAME = "Breadth-First Search"
ALGO_CATEGORY = "Uninformed"
DEFAULT_CONFIG = {"diagonal": False}


def find_path(
    grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    config: dict,
    visit_callback: Optional[Callable] = None,
    yield_steps: bool = False
):
    """
    Find path using Breadth-First Search.
    
    Args:
        grid: Grid object with neighbors() and is_walkable() methods
        start: (row, col) start position
        goal: (row, col) goal position
        config: Configuration dict with 'diagonal' key
        visit_callback: Optional callback(cell, state, info)
        yield_steps: If True, yield after each step
        
    Returns:
        (success, path, metadata) tuple
    """
    start_time = time.time()
    diagonal = config.get("diagonal", False)
    
    queue = deque([start])
    visited = {start}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    nodes_expanded = 0
    nodes_generated = 1
    
    if visit_callback:
        visit_callback(start, 'open', {})
    
    if yield_steps:
        yield
    
    while queue:
        current = queue.popleft()
        nodes_expanded += 1
        
        if visit_callback:
            visit_callback(current, 'closed', {})
        
        if yield_steps:
            yield
        
        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            
            elapsed = time.time() - start_time
            result = (True, path, {
                "nodes_expanded": nodes_expanded,
                "nodes_generated": nodes_generated,
                "path_cost": len(path) - 1,
                "elapsed_time": elapsed
            })
            if yield_steps:
                yield result
            else:
                return result
        
        # Explore neighbors
        for neighbor in grid.neighbors(current[0], current[1], diagonal=diagonal):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                nodes_generated += 1
                
                if visit_callback:
                    visit_callback(neighbor, 'open', {})
                
                if yield_steps:
                    yield
    
    # No path found
    elapsed = time.time() - start_time
    result = (False, None, {
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "path_cost": None,
        "elapsed_time": elapsed
    })
    if yield_steps:
        yield result
    else:
        return result

