"""
Greedy Best-First Search (GBFS) algorithm implementation.
"""

import time
import heapq
from typing import Tuple, List, Optional, Dict, Callable
from .utils import get_heuristic

ALGO_NAME = "Greedy Best-First Search"
ALGO_CATEGORY = "Informed"
DEFAULT_CONFIG = {"diagonal": False, "heuristic": "manhattan"}


def find_path(
    grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    config: dict,
    visit_callback: Optional[Callable] = None,
    yield_steps: bool = False
):
    """
    Find path using Greedy Best-First Search.
    
    Args:
        grid: Grid object with neighbors() and is_walkable() methods
        start: (row, col) start position
        goal: (row, col) goal position
        config: Configuration dict with 'diagonal' and 'heuristic' keys
        visit_callback: Optional callback(cell, state, info)
        yield_steps: If True, yield after each step
        
    Returns:
        (success, path, metadata) tuple
    """
    start_time = time.time()
    diagonal = config.get("diagonal", False)
    heuristic = get_heuristic(config.get("heuristic", "manhattan"))
    
    # Priority queue: (heuristic, row, col)
    pq = [(heuristic(start, goal), start[0], start[1])]
    visited = {start}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    nodes_expanded = 0
    nodes_generated = 1
    
    if visit_callback:
        h_val = heuristic(start, goal)
        visit_callback(start, 'open', {'h': h_val})
    
    if yield_steps:
        yield
    
    while pq:
        _, r, c = heapq.heappop(pq)
        current = (r, c)
        nodes_expanded += 1
        
        if visit_callback:
            h_val = heuristic(current, goal)
            visit_callback(current, 'closed', {'h': h_val})
        
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
                h_val = heuristic(neighbor, goal)
                heapq.heappush(pq, (h_val, neighbor[0], neighbor[1]))
                nodes_generated += 1
                
                if visit_callback:
                    visit_callback(neighbor, 'open', {'h': h_val})
                
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

