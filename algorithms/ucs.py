"""
Uniform Cost Search (UCS) algorithm implementation.
"""

import time
import heapq
from typing import Tuple, List, Optional, Dict, Callable

ALGO_NAME = "Uniform Cost Search"
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
    Find path using Uniform Cost Search (Dijkstra's algorithm).
    
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
    
    # Priority queue: (cost, row, col)
    pq = [(0, start[0], start[1])]
    g_scores: Dict[Tuple[int, int], float] = {start: 0}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    visited = set()
    nodes_expanded = 0
    nodes_generated = 1
    
    if visit_callback:
        visit_callback(start, 'open', {'g': 0})
    
    if yield_steps:
        yield
    
    while pq:
        cost, r, c = heapq.heappop(pq)
        current = (r, c)
        
        if current in visited:
            continue
        
        visited.add(current)
        nodes_expanded += 1
        
        if visit_callback:
            visit_callback(current, 'closed', {'g': cost})
        
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
                "path_cost": cost,
                "elapsed_time": elapsed
            })
            if yield_steps:
                yield result
            else:
                return result
        
        # Explore neighbors
        for neighbor in grid.neighbors(current[0], current[1], diagonal=diagonal):
            # Calculate edge cost (1 for cardinal, sqrt(2) for diagonal)
            dr = abs(neighbor[0] - current[0])
            dc = abs(neighbor[1] - current[1])
            edge_cost = 1.414 if (dr == 1 and dc == 1) else 1.0
            
            new_cost = cost + edge_cost
            
            if neighbor not in g_scores or new_cost < g_scores[neighbor]:
                g_scores[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor[0], neighbor[1]))
                nodes_generated += 1
                
                if visit_callback:
                    visit_callback(neighbor, 'open', {'g': new_cost})
                
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

