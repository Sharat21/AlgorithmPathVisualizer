"""
A* Search algorithm implementation.
"""

import time
import heapq
from typing import Tuple, List, Optional, Dict, Callable
from .utils import get_heuristic

ALGO_NAME = "A* Search"
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
    Find path using A* Search.
    
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
    
    # Priority queue: (f_score, row, col)
    g_scores: Dict[Tuple[int, int], float] = {start: 0}
    h_scores: Dict[Tuple[int, int], float] = {start: heuristic(start, goal)}
    f_scores: Dict[Tuple[int, int], float] = {start: h_scores[start]}
    
    pq = [(f_scores[start], start[0], start[1])]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    visited = set()
    nodes_expanded = 0
    nodes_generated = 1
    
    if visit_callback:
        visit_callback(start, 'open', {
            'g': 0,
            'h': h_scores[start],
            'f': f_scores[start]
        })
    
    if yield_steps:
        yield
    
    while pq:
        _, r, c = heapq.heappop(pq)
        current = (r, c)
        
        if current in visited:
            continue
        
        visited.add(current)
        nodes_expanded += 1
        
        if visit_callback:
            visit_callback(current, 'closed', {
                'g': g_scores[current],
                'h': h_scores.get(current, heuristic(current, goal)),
                'f': f_scores[current]
            })
        
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
                "path_cost": g_scores[current],
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
            
            tentative_g = g_scores[current] + edge_cost
            
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                h_scores[neighbor] = heuristic(neighbor, goal)
                f_scores[neighbor] = tentative_g + h_scores[neighbor]
                came_from[neighbor] = current
                heapq.heappush(pq, (f_scores[neighbor], neighbor[0], neighbor[1]))
                nodes_generated += 1
                
                if visit_callback:
                    visit_callback(neighbor, 'open', {
                        'g': tentative_g,
                        'h': h_scores[neighbor],
                        'f': f_scores[neighbor]
                    })
                
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

