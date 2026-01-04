"""
Depth-First Search (DFS) algorithm implementation.
"""

import time
from typing import Tuple, List, Optional, Dict, Callable, Iterator
from collections import deque

ALGO_NAME = "Depth-First Search"
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
    Find path using Depth-First Search.
    
    Args:
        grid: Grid object with neighbors() and is_walkable() methods
        start: (row, col) start position
        goal: (row, col) goal position
        config: Configuration dict with 'diagonal' key
        visit_callback: Optional callback(cell, state, info)
        yield_steps: If True, yield after each step (returns generator)
        
    Returns:
        Generator if yield_steps=True, otherwise (success, path, metadata) tuple
    """
    start_time = time.time()
    diagonal = config.get("diagonal", False)
    
    stack = deque([start])
    visited = {start}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    nodes_expanded = 0
    nodes_generated = 1
    
    if visit_callback:
        visit_callback(start, 'open', {})
    
    if yield_steps:
        yield
    
    while stack:
        current = stack.pop()
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
                stack.append(neighbor)
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


# Generator wrapper for yield_steps mode
def find_path_generator(*args, **kwargs):
    """Generator wrapper that always yields steps."""
    kwargs['yield_steps'] = True
    gen = find_path(*args, **kwargs)
    if isinstance(gen, tuple):
        # Not a generator, return as-is
        return gen
    return gen

