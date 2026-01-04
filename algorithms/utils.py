"""
Utility functions for pathfinding algorithms, including heuristics.
"""

from typing import Tuple


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate Manhattan distance between two positions.
    
    Args:
        pos1: (row, col) tuple
        pos2: (row, col) tuple
        
    Returns:
        Manhattan distance
    """
    r1, c1 = pos1
    r2, c2 = pos2
    return abs(r1 - r2) + abs(c1 - c2)


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two positions.
    
    Args:
        pos1: (row, col) tuple
        pos2: (row, col) tuple
        
    Returns:
        Euclidean distance
    """
    r1, c1 = pos1
    r2, c2 = pos2
    return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5


def chebyshev_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate Chebyshev distance (max of row/col differences).
    Useful for diagonal movement.
    
    Args:
        pos1: (row, col) tuple
        pos2: (row, col) tuple
        
    Returns:
        Chebyshev distance
    """
    r1, c1 = pos1
    r2, c2 = pos2
    return max(abs(r1 - r2), abs(c1 - c2))


def get_heuristic(heuristic_name: str):
    """
    Get heuristic function by name.
    
    Args:
        heuristic_name: Name of heuristic ('manhattan', 'euclidean', 'chebyshev')
        
    Returns:
        Heuristic function
    """
    heuristics = {
        'manhattan': manhattan_distance,
        'euclidean': euclidean_distance,
        'chebyshev': chebyshev_distance,
    }
    return heuristics.get(heuristic_name.lower(), manhattan_distance)

