"""
Iterative Deepening A* (IDA*) algorithm implementation.
"""

import time
import math
from typing import Tuple, List, Optional, Dict, Callable
from .utils import get_heuristic

ALGO_NAME = "Iterative Deepening A*"
ALGO_CATEGORY = "Informed"
DEFAULT_CONFIG = {"diagonal": False, "heuristic": "manhattan"}


def _search(
    grid,
    node: Tuple[int, int],
    goal: Tuple[int, int],
    g: float,
    threshold: float,
    diagonal: bool,
    heuristic,
    path: List[Tuple[int, int]],
    visit_callback: Optional[Callable],
    yield_steps: bool,
    nodes_expanded: List[int],
    nodes_generated: List[int]
):
    """
    Recursive search helper for IDA*.
    Returns generator if yield_steps=True, otherwise returns tuple.
    """
    h = heuristic(node, goal)
    f = g + h
    
    if f > threshold:
        if yield_steps:
            yield (f, None)
        else:
            return (f, None)
    
    if node == goal:
        result_path = path + [node]
        if yield_steps:
            yield (f, result_path)
        else:
            return (f, result_path)
    
    min_threshold = math.inf
    
    nodes_expanded[0] += 1
    
    if visit_callback:
        visit_callback(node, 'closed', {'g': g, 'h': h, 'f': f})
    
    if yield_steps:
        yield None  # Yield for visualization
    
    # Explore neighbors
    for neighbor in grid.neighbors(node[0], node[1], diagonal=diagonal):
        if neighbor in path:
            continue
        
        # Calculate edge cost
        dr = abs(neighbor[0] - node[0])
        dc = abs(neighbor[1] - node[1])
        edge_cost = 1.414 if (dr == 1 and dc == 1) else 1.0
        
        nodes_generated[0] += 1
        
        if visit_callback:
            neighbor_g = g + edge_cost
            neighbor_h = heuristic(neighbor, goal)
            neighbor_f = neighbor_g + neighbor_h
            visit_callback(neighbor, 'open', {
                'g': neighbor_g,
                'h': neighbor_h,
                'f': neighbor_f
            })
        
        if yield_steps:
            yield None  # Yield for visualization
        
        # Recursive search
        if yield_steps:
            # Generator case
            gen = _search(
                grid, neighbor, goal, g + edge_cost, threshold,
                diagonal, heuristic, path + [node],
                visit_callback, yield_steps, nodes_expanded, nodes_generated
            )
            for result in gen:
                if result is None:
                    yield None
                else:
                    next_threshold, found_path = result
                    if found_path is not None:
                        yield (next_threshold, found_path)
                        return
                    if next_threshold < min_threshold:
                        min_threshold = next_threshold
        else:
            # Non-generator case
            result = _search(
                grid, neighbor, goal, g + edge_cost, threshold,
                diagonal, heuristic, path + [node],
                visit_callback, yield_steps, nodes_expanded, nodes_generated
            )
            next_threshold, found_path = result
            if found_path is not None:
                if yield_steps:
                    yield (next_threshold, found_path)
                else:
                    return (next_threshold, found_path)
            if next_threshold < min_threshold:
                min_threshold = next_threshold
    
    if yield_steps:
        yield (min_threshold, None)
    else:
        return (min_threshold, None)


def find_path(
    grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    config: dict,
    visit_callback: Optional[Callable] = None,
    yield_steps: bool = False
):
    """
    Find path using Iterative Deepening A*.
    
    Args:
        grid: Grid object with neighbors() and is_walkable() methods
        start: (row, col) start position
        goal: (row, col) goal position
        config: Configuration dict with 'diagonal' and 'heuristic' keys
        visit_callback: Optional callback(cell, state, info)
        yield_steps: If True, yield after each step
        
    Returns:
        Generator if yield_steps=True, otherwise (success, path, metadata) tuple
    """
    start_time = time.time()
    diagonal = config.get("diagonal", False)
    heuristic = get_heuristic(config.get("heuristic", "manhattan"))
    
    threshold = heuristic(start, goal)
    nodes_expanded = [0]
    nodes_generated = [1]
    
    if visit_callback:
        h_val = heuristic(start, goal)
        visit_callback(start, 'open', {'g': 0, 'h': h_val, 'f': h_val})
    
    if yield_steps:
        yield
    
    max_iterations = grid.rows * grid.cols * 10  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        if yield_steps:
            # Generator mode
            gen = _search(
                grid, start, goal, 0, threshold, diagonal, heuristic, [],
                visit_callback, yield_steps, nodes_expanded, nodes_generated
            )
            final_result = None
            for result in gen:
                if result is None:
                    yield  # Visualization step
                else:
                    final_result = result
            
            if final_result:
                next_threshold, path = final_result
            else:
                next_threshold, path = (math.inf, None)
        else:
            # Non-generator mode
            result = _search(
                grid, start, goal, 0, threshold, diagonal, heuristic, [],
                visit_callback, yield_steps, nodes_expanded, nodes_generated
            )
            next_threshold, path = result
        
        if path is not None:
            elapsed = time.time() - start_time
            result_tuple = (True, path, {
                "nodes_expanded": nodes_expanded[0],
                "nodes_generated": nodes_generated[0],
                "path_cost": len(path) - 1,
                "elapsed_time": elapsed
            })
            if yield_steps:
                yield result_tuple
            else:
                return result_tuple
        
        if next_threshold == math.inf:
            break
        
        threshold = next_threshold
        iteration += 1
    
    # No path found
    elapsed = time.time() - start_time
    result_tuple = (False, None, {
        "nodes_expanded": nodes_expanded[0],
        "nodes_generated": nodes_generated[0],
        "path_cost": None,
        "elapsed_time": elapsed
    })
    if yield_steps:
        yield result_tuple
    else:
        return result_tuple

