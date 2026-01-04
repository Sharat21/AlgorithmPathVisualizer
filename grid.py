"""
Grid and cell management, drawing primitives, and grid utilities.

This module provides the Grid class which handles all grid-level operations,
visualization, and cell state management. Algorithms interact with the grid
through a clean API without needing to know about visualization details.
"""

import tkinter as tk
from typing import Tuple, Iterator, List, Optional, Callable
from enum import Enum


class CellState(Enum):
    """Cell state enumeration."""
    EMPTY = "EMPTY"
    WALL = "WALL"
    START = "START"
    END = "END"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PATH = "PATH"


# Color constants (RGB tuples for tkinter)
COLORS = {
    CellState.EMPTY: "#E0E0E0",  # light gray
    CellState.WALL: "#2C2C2C",   # dark gray/black
    CellState.START: "#2196F3",  # blue
    CellState.END: "#F44336",    # red
    CellState.OPEN: "#FFEB3B",   # yellow
    CellState.CLOSED: "#FF9800", # orange
    CellState.PATH: "#4CAF50",   # green
}


class Grid:
    """
    Grid class for managing cell states and visualization.
    
    Handles all grid-level operations including:
    - Cell state management
    - Visualization on Tkinter Canvas
    - Neighbor queries
    - Path highlighting
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        canvas: tk.Canvas,
        cell_size: int = 16,
        sound_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the grid.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            canvas: Tkinter Canvas widget for drawing
            cell_size: Size of each cell in pixels
            sound_callback: Optional callback function for sound effects
        """
        self.rows = rows
        self.cols = cols
        self.canvas = canvas
        self.cell_size = cell_size
        self.sound_callback = sound_callback
        
        # Internal state: grid[row][col] = CellState
        self.grid = [[CellState.EMPTY for _ in range(cols)] for _ in range(rows)]
        
        # Track start and end positions
        self.start_pos: Optional[Tuple[int, int]] = None
        self.end_pos: Optional[Tuple[int, int]] = None
        
        # Canvas rectangles for each cell (for efficient updates)
        self.rects = {}
        self.labels = {}  # For g/h/f values
        
        # Draw initial grid
        self._draw_grid()
    
    def _draw_grid(self) -> None:
        """Draw the initial grid on the canvas."""
        self.canvas.delete("all")
        self.rects = {}
        self.labels = {}
        
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Create rectangle
                rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=COLORS[CellState.EMPTY],
                    outline="#CCCCCC",
                    width=1,
                    tags=f"cell_{r}_{c}"
                )
                self.rects[(r, c)] = rect_id
    
    def reset(self, keep_start_end: bool = True) -> None:
        """
        Reset grid to initial empty state.
        
        Args:
            keep_start_end: If True, preserve start and end positions
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != CellState.WALL:
                    self.grid[r][c] = CellState.EMPTY
        
        # Restore start and end if requested
        if keep_start_end:
            if self.start_pos:
                r, c = self.start_pos
                self.grid[r][c] = CellState.START
            if self.end_pos:
                r, c = self.end_pos
                self.grid[r][c] = CellState.END
        
        self._update_visualization()
    
    def clear_all(self) -> None:
        """Clear everything including start/end and walls."""
        self.start_pos = None
        self.end_pos = None
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = CellState.EMPTY
        self._update_visualization()
    
    def set_cell(self, r: int, c: int, state: CellState) -> None:
        """
        Set a cell state.
        
        Args:
            r: Row index
            c: Column index
            state: CellState enum value
        """
        if not self.in_bounds(r, c):
            return
        
        # Protect start/end from being overwritten (except by explicit set_start/set_end)
        if self.grid[r][c] == CellState.START and state != CellState.START:
            return
        if self.grid[r][c] == CellState.END and state != CellState.END:
            return
        
        self.grid[r][c] = state
        self._draw_cell(r, c)
    
    def get_cell(self, r: int, c: int) -> CellState:
        """
        Get cell state.
        
        Args:
            r: Row index
            c: Column index
            
        Returns:
            CellState enum value
        """
        if not self.in_bounds(r, c):
            return CellState.WALL  # Out of bounds treated as wall
        return self.grid[r][c]
    
    def toggle_wall(self, r: int, c: int) -> None:
        """
        Toggle a wall at the given cell.
        Protects start/end cells from becoming walls.
        
        Args:
            r: Row index
            c: Column index
        """
        if not self.in_bounds(r, c):
            return
        
        # Protect start and end
        if self.grid[r][c] == CellState.START or self.grid[r][c] == CellState.END:
            return
        
        if self.grid[r][c] == CellState.WALL:
            self.grid[r][c] = CellState.EMPTY
        else:
            self.grid[r][c] = CellState.WALL
        
        self._draw_cell(r, c)
    
    def neighbors(
        self,
        r: int,
        c: int,
        diagonal: bool = False
    ) -> Iterator[Tuple[int, int]]:
        """
        Get walkable neighbors of a cell.
        
        Args:
            r: Row index
            c: Column index
            diagonal: If True, include diagonal neighbors
            
        Yields:
            (row, col) tuples of walkable neighbors
        """
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1)  # 4-directional
        ]
        
        if diagonal:
            directions.extend([
                (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonal
            ])
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.is_walkable(nr, nc):
                yield (nr, nc)
    
    def set_start(self, r: int, c: int) -> None:
        """
        Set the start position.
        
        Args:
            r: Row index
            c: Column index
        """
        if not self.in_bounds(r, c):
            return
        
        # Clear old start
        if self.start_pos:
            old_r, old_c = self.start_pos
            if self.grid[old_r][old_c] == CellState.START:
                self.grid[old_r][old_c] = CellState.EMPTY
                self._draw_cell(old_r, old_c)
        
        # Set new start
        self.start_pos = (r, c)
        self.grid[r][c] = CellState.START
        self._draw_cell(r, c)
    
    def set_end(self, r: int, c: int) -> None:
        """
        Set the end position.
        
        Args:
            r: Row index
            c: Column index
        """
        if not self.in_bounds(r, c):
            return
        
        # Clear old end
        if self.end_pos:
            old_r, old_c = self.end_pos
            if self.grid[old_r][old_c] == CellState.END:
                self.grid[old_r][old_c] = CellState.EMPTY
                self._draw_cell(old_r, old_c)
        
        # Set new end
        self.end_pos = (r, c)
        self.grid[r][c] = CellState.END
        self._draw_cell(r, c)
    
    def draw_cell_label(self, r: int, c: int, text: str) -> None:
        """
        Draw text label on a cell (for g/h/f values).
        
        Args:
            r: Row index
            c: Column index
            text: Text to display
        """
        if not self.in_bounds(r, c):
            return
        
        # Remove old label if exists
        if (r, c) in self.labels:
            self.canvas.delete(self.labels[(r, c)])
        
        if text:
            x = c * self.cell_size + self.cell_size // 2
            y = r * self.cell_size + self.cell_size // 2
            
            label_id = self.canvas.create_text(
                x, y,
                text=text,
                fill="black",
                font=("Arial", max(6, self.cell_size // 4)),
                tags=f"label_{r}_{c}"
            )
            self.labels[(r, c)] = label_id
    
    def mark_open(self, r: int, c: int, info: Optional[dict] = None) -> None:
        """
        Mark a cell as open (in frontier).
        
        Args:
            r: Row index
            c: Column index
            info: Optional metadata (e.g., g, h, f values)
        """
        if not self.in_bounds(r, c):
            return
        
        # Don't overwrite start/end
        if self.grid[r][c] in (CellState.START, CellState.END):
            return
        
        self.grid[r][c] = CellState.OPEN
        self._draw_cell(r, c)
        
        # Show g/h/f if provided
        if info and self.cell_size >= 12:
            g = info.get('g', '')
            h = info.get('h', '')
            f = info.get('f', '')
            if g != '' or h != '' or f != '':
                label = f"g:{g}\nh:{h}\nf:{f}" if f != '' else f"g:{g}"
                self.draw_cell_label(r, c, label)
        
        # Play sound
        if self.sound_callback:
            self.sound_callback("visit")
    
    def mark_closed(self, r: int, c: int, info: Optional[dict] = None) -> None:
        """
        Mark a cell as closed (visited/expanded).
        
        Args:
            r: Row index
            c: Column index
            info: Optional metadata
        """
        if not self.in_bounds(r, c):
            return
        
        # Don't overwrite start/end
        if self.grid[r][c] in (CellState.START, CellState.END):
            return
        
        self.grid[r][c] = CellState.CLOSED
        self._draw_cell(r, c)
        
        # Update label if info provided
        if info and self.cell_size >= 12:
            g = info.get('g', '')
            h = info.get('h', '')
            f = info.get('f', '')
            if g != '' or h != '' or f != '':
                label = f"g:{g}\nh:{h}\nf:{f}" if f != '' else f"g:{g}"
                self.draw_cell_label(r, c, label)
    
    def mark_path(self, cells: List[Tuple[int, int]]) -> None:
        """
        Mark a list of cells as the final path.
        
        Args:
            cells: List of (row, col) tuples from start to end
        """
        for r, c in cells:
            # Don't overwrite start/end
            if self.grid[r][c] not in (CellState.START, CellState.END):
                self.grid[r][c] = CellState.PATH
                self._draw_cell(r, c)
                
                # Play path sound
                if self.sound_callback:
                    self.sound_callback("path")
    
    def in_bounds(self, r: int, c: int) -> bool:
        """
        Check if coordinates are within grid bounds.
        
        Args:
            r: Row index
            c: Column index
            
        Returns:
            True if in bounds, False otherwise
        """
        return 0 <= r < self.rows and 0 <= c < self.cols
    
    def is_walkable(self, r: int, c: int) -> bool:
        """
        Check if a cell is walkable (not a wall and in bounds).
        
        Args:
            r: Row index
            c: Column index
            
        Returns:
            True if walkable, False otherwise
        """
        if not self.in_bounds(r, c):
            return False
        return self.grid[r][c] != CellState.WALL
    
    def _draw_cell(self, r: int, c: int) -> None:
        """Update visualization for a single cell."""
        if not self.in_bounds(r, c):
            return
        
        state = self.grid[r][c]
        color = COLORS[state]
        
        if (r, c) in self.rects:
            self.canvas.itemconfig(self.rects[(r, c)], fill=color)
    
    def _update_visualization(self) -> None:
        """Update visualization for all cells."""
        for r in range(self.rows):
            for c in range(self.cols):
                self._draw_cell(r, c)
    
    def get_cell_at_pixel(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """
        Convert pixel coordinates to grid cell coordinates.
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            
        Returns:
            (row, col) tuple or None if out of bounds
        """
        c = x // self.cell_size
        r = y // self.cell_size
        
        if self.in_bounds(r, c):
            return (r, c)
        return None
    
    def randomize_barricades(self, density: float) -> None:
        """
        Randomly place walls according to density.
        Never places walls on start/end cells.
        
        Args:
            density: Fraction of cells to fill with walls (0.0 to 1.0)
        """
        import random
        
        # Clear existing walls (but keep start/end)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == CellState.WALL:
                    self.grid[r][c] = CellState.EMPTY
        
        # Place random walls
        total_cells = self.rows * self.cols
        num_walls = int(total_cells * density)
        
        # Generate candidate positions (excluding start/end)
        candidates = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] not in (CellState.START, CellState.END):
                    candidates.append((r, c))
        
        # Randomly select positions
        random.shuffle(candidates)
        for i in range(min(num_walls, len(candidates))):
            r, c = candidates[i]
            self.grid[r][c] = CellState.WALL
        
        self._update_visualization()

