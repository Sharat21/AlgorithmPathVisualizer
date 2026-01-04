"""
Iterative Deepening Search (IDS) algorithm implementation.
"""

import time
from typing import Tuple, List, Optional, Dict, Callable
from collections import deque

ALGO_NAME = "Iterative Deepening Search"
ALGO_CATEGORY = "Uninformed"
DEFAULT_CONFIG = {"diagonal": False}


def _dls(
    grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    depth_limit: int,
    diagonal: bool,
    visit_callback: Optional[Callable],
    yield_steps: bool,
    visited_at_depth: Dict[Tuple[int, int], int]
):
    """
    Depth-Limited Search helper function.
    
    Returns:
        (found, path, nodes_expanded) tuple
    """
    stack = deque([(start, 0)])
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    nodes_expanded = 0
    
    if visit_callback:
        visit_callback(start, 'open', {})
    
    if yield_steps:
        yield
    
    while stack:
        current, depth = stack.pop()
        
        if depth > depth_limit:
            continue
        
        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            result = (True, path, nodes_expanded)
            if yield_steps:
                yield result
            else:
                return result
        
        # Only expand if not visited at this depth or shallower
        if current in visited_at_depth and visited_at_depth[current] <= depth:
            continue
        
        visited_at_depth[current] = depth
        nodes_expanded += 1
        
        if visit_callback:
            visit_callback(current, 'closed', {})
        
        if yield_steps:
            yield
        
        if depth < depth_limit:
            # Explore neighbors
            for neighbor in grid.neighbors(current[0], current[1], diagonal=diagonal):
                if neighbor not in visited_at_depth or visited_at_depth[neighbor] > depth + 1:
                    came_from[neighbor] = current
                    stack.append((neighbor, depth + 1))
                    
                    if visit_callback:
                        visit_callback(neighbor, 'open', {})
                    
                    if yield_steps:
                        yield
    
    result = (False, None, nodes_expanded)
    if yield_steps:
        yield result
    else:
        return result


def find_path(
    grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    config: dict,
    visit_callback: Optional[Callable] = None,
    yield_steps: bool = False
):
    """
    Find path using Iterative Deepening Search.
    
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
    
    nodes_expanded_total = 0
    nodes_generated = 1
    max_depth = grid.rows * grid.cols  # Upper bound
    
    for depth_limit in range(max_depth):
        visited_at_depth: Dict[Tuple[int, int], int] = {}
        
        # Create generator for DLS
        dls_gen = _dls(
            grid, start, goal, depth_limit, diagonal,
            visit_callback, yield_steps, visited_at_depth
        )
        
        if yield_steps:
            # Consume generator
            result = None
            try:
                while True:
                    result = next(dls_gen)
                    yield
            except StopIteration:
                found, path, nodes_expanded = result if result else (False, None, 0)
        else:
            # Run to completion
            found, path, nodes_expanded = _dls(
                grid, start, goal, depth_limit, diagonal,
                visit_callback, False, visited_at_depth
            )
        
        nodes_expanded_total += nodes_expanded
        
        if found:
            elapsed = time.time() - start_time
            result = (True, path, {
                "nodes_expanded": nodes_expanded_total,
                "nodes_generated": nodes_generated,
                "path_cost": len(path) - 1 if path else 0,
                "elapsed_time": elapsed
            })
            if yield_steps:
                yield result
            else:
                return result
    
    # No path found
    elapsed = time.time() - start_time
    result = (False, None, {
        "nodes_expanded": nodes_expanded_total,
        "nodes_generated": nodes_generated,
        "path_cost": None,
        "elapsed_time": elapsed
    })
    if yield_steps:
        yield result
    else:
        return result

