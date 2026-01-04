"""
Main application entry point for Algorithm Path Visualizer.

This module provides the Tkinter GUI and orchestrates algorithm execution,
visualization, and user interactions.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from typing import Optional, Dict, Any, Tuple
from grid import Grid, CellState
from algorithms import discover_algorithms


class SoundPlayer:
    """Simple sound player with fallback support."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._player = None
        self._try_load_player()
    
    def _try_load_player(self):
        """Try to load an audio library."""
        # Try pygame first
        try:
            import pygame
            pygame.mixer.init()
            self._player = "pygame"
            return
        except ImportError:
            pass
        
        # Try simpleaudio
        try:
            import simpleaudio as sa
            self._player = "simpleaudio"
            return
        except ImportError:
            pass
        
        # Try winsound (Windows only)
        try:
            import winsound
            self._player = "winsound"
            return
        except ImportError:
            pass
        
        # No audio library available
        self._player = None
    
    def play(self, sound_type: str = "visit"):
        """Play a sound effect."""
        if not self.enabled or self._player is None:
            return
        
        try:
            if self._player == "pygame":
                # Generate a simple tone
                try:
                    import pygame
                    import numpy as np
                    sample_rate = 44100
                    duration = 0.1
                    if sound_type == "visit":
                        frequency = 440
                    elif sound_type == "path":
                        frequency = 880
                    else:
                        frequency = 440
                    
                    frames = int(duration * sample_rate)
                    arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
                    arr = (arr * 32767).astype(np.int16)
                    sound = pygame.sndarray.make_sound(arr)
                    sound.play()
                except ImportError:
                    # numpy not available, skip sound
                    pass
            elif self._player == "simpleaudio":
                # Similar tone generation
                try:
                    import simpleaudio as sa
                    import numpy as np
                    sample_rate = 44100
                    duration = 0.1
                    if sound_type == "visit":
                        frequency = 440
                    elif sound_type == "path":
                        frequency = 880
                    else:
                        frequency = 440
                    
                    frames = int(duration * sample_rate)
                    arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
                    arr = (arr * 32767).astype(np.int16)
                    play_obj = sa.play_buffer(arr, 1, 2, sample_rate)
                except ImportError:
                    # numpy not available, skip sound
                    pass
            elif self._player == "winsound":
                import winsound
                if sound_type == "visit":
                    winsound.Beep(440, 50)
                elif sound_type == "path":
                    winsound.Beep(880, 100)
        except Exception:
            # Silently fail if sound playback fails
            pass


class AlgorithmVisualizer:
    """Main application class."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Algorithm Path Visualizer")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize state
        self.grid: Optional[Grid] = None
        self.current_algorithm = None
        self.algorithm_generator = None
        self.is_running = False
        self.is_paused = False
        self.mode = "select_start"  # select_start, select_end, toggle_walls
        self.animation_id = None
        
        # Sound player
        self.sound_player = SoundPlayer(self.config.get("sound_enabled", True))
        
        # Discover algorithms
        self.algorithms = discover_algorithms()
        self.algorithm_modules = {algo["name"]: algo for algo in self.algorithms}
        
        # Create UI
        self._create_ui()
        
        # Initialize grid immediately - will be updated when window is shown
        # We'll also schedule a refresh after window is displayed
        self._init_grid()
        self.root.after(200, self._refresh_grid)
    
    def _load_config(self) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "grid_size": 40,
            "cell_size": 16,
            "animation_speed": 10,  # milliseconds
            "sound_enabled": True,
            "random_density": 0.3,
            "show_exploration": True,
            "show_values": False
        }
        
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception:
                pass
        
        return default_config
    
    def _save_config(self):
        """Save current configuration to file."""
        config_path = "config.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass
    
    def _create_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Algorithm selection
        ttk.Label(control_frame, text="Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.algorithm_var = tk.StringVar()
        algo_combo = ttk.Combobox(control_frame, textvariable=self.algorithm_var, state="readonly", width=25)
        algo_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Populate algorithms by category
        algo_list = []
        uninformed = [a["name"] for a in self.algorithms if a["category"] == "Uninformed"]
        informed = [a["name"] for a in self.algorithms if a["category"] == "Informed"]
        
        if uninformed:
            algo_list.extend([f"--- Uninformed ---"] + uninformed)
        if informed:
            algo_list.extend([f"--- Informed ---"] + informed)
        
        algo_combo['values'] = algo_list
        if algo_list:
            # Select first actual algorithm (skip separator)
            for item in algo_list:
                if not item.startswith("---"):
                    self.algorithm_var.set(item)
                    break
        algo_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)
        
        # Grid setup
        ttk.Label(control_frame, text="Grid Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.grid_size_var = tk.IntVar(value=self.config.get("grid_size", 40))
        grid_size_scale = ttk.Scale(control_frame, from_=10, to=80, variable=self.grid_size_var, orient=tk.HORIZONTAL)
        grid_size_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.grid_size_label = ttk.Label(control_frame, text=str(self.grid_size_var.get()))
        self.grid_size_label.grid(row=1, column=2, padx=5)
        grid_size_scale.configure(command=lambda v: self.grid_size_label.config(text=str(int(float(v)))))
        grid_size_scale.bind("<ButtonRelease-1>", lambda e: self._on_grid_size_change())
        
        # Animation speed
        ttk.Label(control_frame, text="Speed (ms):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.speed_var = tk.IntVar(value=self.config.get("animation_speed", 10))
        speed_scale = ttk.Scale(control_frame, from_=1, to=100, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        self.speed_label = ttk.Label(control_frame, text=str(self.speed_var.get()))
        self.speed_label.grid(row=2, column=2, padx=5)
        speed_scale.configure(command=lambda v: self.speed_label.config(text=str(int(float(v)))))
        
        # Random density
        ttk.Label(control_frame, text="Wall Density:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.density_var = tk.DoubleVar(value=self.config.get("random_density", 0.3))
        density_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, variable=self.density_var, orient=tk.HORIZONTAL)
        density_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        self.density_label = ttk.Label(control_frame, text=f"{self.density_var.get():.2f}")
        self.density_label.grid(row=3, column=2, padx=5)
        density_scale.configure(command=lambda v: self.density_label.config(text=f"{float(v):.2f}"))
        
        # Checkboxes
        self.sound_var = tk.BooleanVar(value=self.config.get("sound_enabled", True))
        ttk.Checkbutton(control_frame, text="Sound", variable=self.sound_var,
                       command=self._on_sound_toggle).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        self.show_exploration_var = tk.BooleanVar(value=self.config.get("show_exploration", True))
        ttk.Checkbutton(control_frame, text="Show Exploration", variable=self.show_exploration_var).grid(
            row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        self.show_values_var = tk.BooleanVar(value=self.config.get("show_values", False))
        ttk.Checkbutton(control_frame, text="Show g/h/f Values", variable=self.show_values_var).grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Buttons - Grid setup
        button_frame1 = ttk.LabelFrame(control_frame, text="Grid Setup", padding="5")
        button_frame1.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(button_frame1, text="Select Start", command=self._select_start_mode).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Button(button_frame1, text="Select End", command=self._select_end_mode).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Button(button_frame1, text="Toggle Walls", command=self._toggle_walls_mode).grid(
            row=1, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Button(button_frame1, text="Randomize", command=self._randomize_walls).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Button(button_frame1, text="Clear Grid", command=self._clear_grid).grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=2, pady=2)
        
        # Buttons - Algorithm control
        button_frame2 = ttk.LabelFrame(control_frame, text="Algorithm Control", padding="5")
        button_frame2.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(button_frame2, text="Start", command=self._start_algorithm).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Button(button_frame2, text="Pause", command=self._pause_algorithm).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Button(button_frame2, text="Step", command=self._step_algorithm).grid(
            row=1, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
        ttk.Button(button_frame2, text="Stop", command=self._stop_algorithm).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=2, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Right panel - canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", width=640, height=640, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
    
    def _init_grid(self):
        """Initialize the grid."""
        if self.is_running:
            return  # Don't reinitialize while running
        
        grid_size = self.grid_size_var.get()
        cell_size = self.config.get("cell_size", 16)
        
        # Force canvas to update and get proper dimensions
        self.root.update()
        self.root.update_idletasks()
        self.canvas.update_idletasks()
        
        # Get actual canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # If canvas hasn't been sized yet, use default and configure
        if canvas_width <= 1 or canvas_height <= 1:
            # Set a minimum size for the canvas
            self.canvas.config(width=640, height=640)
            canvas_width = 640
            canvas_height = 640
            self.root.update_idletasks()
        
        # Adjust cell size to fit canvas
        max_cell_size = min(canvas_width // grid_size, canvas_height // grid_size)
        if max_cell_size < 1:
            max_cell_size = 1
        cell_size = min(cell_size, max_cell_size)
        
        # Ensure minimum cell size for visibility
        if cell_size < 4:
            cell_size = 4
        
        # Create grid
        self.grid = Grid(
            grid_size, grid_size, self.canvas, cell_size,
            sound_callback=self._sound_callback
        )
        
        # Force a redraw to ensure grid is visible
        self.canvas.update_idletasks()
        self.root.update_idletasks()
    
    def _refresh_grid(self):
        """Refresh grid after window is fully displayed."""
        if self.grid is not None:
            # Recalculate cell size based on actual canvas dimensions
            grid_size = self.grid_size_var.get()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                max_cell_size = min(canvas_width // grid_size, canvas_height // grid_size)
                if max_cell_size > 0 and max_cell_size != self.grid.cell_size:
                    # Recreate grid with proper cell size
                    self._init_grid()
    
    def _on_grid_size_change(self):
        """Handle grid size change."""
        if self.is_running:
            messagebox.showwarning("Warning", "Stop the algorithm before changing grid size.")
            return
        self._init_grid()
    
    def _on_sound_toggle(self):
        """Handle sound toggle."""
        self.sound_player.enabled = self.sound_var.get()
        self.config["sound_enabled"] = self.sound_var.get()
        self._save_config()
    
    def _on_algorithm_change(self, event=None):
        """Handle algorithm selection change."""
        if self.is_running:
            messagebox.showwarning("Warning", "Stop the algorithm before changing algorithm.")
            return
    
    def _sound_callback(self, sound_type: str):
        """Callback for sound effects."""
        if self.sound_var.get():
            self.sound_player.play(sound_type)
    
    def _select_start_mode(self):
        """Enter start selection mode."""
        self.mode = "select_start"
        self.status_var.set("Click on grid to set start position")
    
    def _select_end_mode(self):
        """Enter end selection mode."""
        self.mode = "select_end"
        self.status_var.set("Click on grid to set end position")
    
    def _toggle_walls_mode(self):
        """Enter wall toggle mode."""
        self.mode = "toggle_walls"
        self.status_var.set("Click on grid to toggle walls")
    
    def _on_canvas_click(self, event):
        """Handle canvas click."""
        if self.grid is None:
            return
        
        cell = self.grid.get_cell_at_pixel(event.x, event.y)
        if cell is None:
            return
        
        r, c = cell
        
        if self.mode == "select_start":
            self.grid.set_start(r, c)
            self.status_var.set(f"Start set at ({r}, {c})")
        elif self.mode == "select_end":
            self.grid.set_end(r, c)
            self.status_var.set(f"End set at ({r}, {c})")
        elif self.mode == "toggle_walls":
            self.grid.toggle_wall(r, c)
    
    def _on_canvas_drag(self, event):
        """Handle canvas drag (for drawing walls)."""
        if self.mode == "toggle_walls":
            self._on_canvas_click(event)
    
    def _randomize_walls(self):
        """Randomize walls on the grid."""
        if self.is_running:
            messagebox.showwarning("Warning", "Stop the algorithm before randomizing walls.")
            return
        density = self.density_var.get()
        self.grid.randomize_barricades(density)
        self.status_var.set(f"Randomized walls with density {density:.2f}")
    
    def _clear_grid(self):
        """Clear the grid."""
        if self.is_running:
            messagebox.showwarning("Warning", "Stop the algorithm before clearing grid.")
            return
        self.grid.clear_all()
        self.status_var.set("Grid cleared")
    
    def _start_algorithm(self):
        """Start algorithm execution."""
        if self.is_running and not self.is_paused:
            return
        
        if self.grid.start_pos is None or self.grid.end_pos is None:
            messagebox.showerror("Error", "Please set both start and end positions.")
            return
        
        algo_name = self.algorithm_var.get()
        if not algo_name or algo_name.startswith("---"):
            messagebox.showerror("Error", "Please select an algorithm.")
            return
        
        if algo_name not in self.algorithm_modules:
            messagebox.showerror("Error", f"Algorithm '{algo_name}' not found.")
            return
        
        algo_info = self.algorithm_modules[algo_name]
        algo_module = algo_info["module"]
        
        # Get algorithm configuration
        config = algo_info["default_config"].copy()
        config["diagonal"] = False  # Can be made configurable
        
        # Create visit callback
        def visit_callback(cell, state, info=None):
            r, c = cell
            if state == 'open':
                self.grid.mark_open(r, c, info if self.show_values_var.get() else None)
            elif state == 'closed':
                self.grid.mark_closed(r, c, info if self.show_values_var.get() else None)
            elif state == 'path':
                # Path marking is handled separately
                pass
        
        # Start algorithm
        self.is_running = True
        self.is_paused = False
        
        try:
            self.algorithm_generator = algo_module.find_path(
                self.grid,
                self.grid.start_pos,
                self.grid.end_pos,
                config,
                visit_callback=visit_callback if self.show_exploration_var.get() else None,
                yield_steps=True
            )
            
            self.status_var.set(f"Running {algo_name}...")
            self._step_algorithm_loop()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start algorithm: {str(e)}")
            self.is_running = False
            self.algorithm_generator = None
    
    def _step_algorithm_loop(self):
        """Step through algorithm generator."""
        if not self.is_running or self.is_paused:
            return
        
        try:
            result = next(self.algorithm_generator)
            
            # Check if we got a final result
            if isinstance(result, tuple) and len(result) == 3:
                success, path, metadata = result
                self._handle_algorithm_complete(success, path, metadata)
                return
            
            # Schedule next step
            delay = self.speed_var.get()
            self.animation_id = self.root.after(delay, self._step_algorithm_loop)
            self.root.update_idletasks()
        except StopIteration:
            # Generator exhausted without result (shouldn't happen)
            self._handle_algorithm_complete(False, None, {})
        except Exception as e:
            messagebox.showerror("Error", f"Algorithm error: {str(e)}")
            self._stop_algorithm()
    
    def _handle_algorithm_complete(self, success: bool, path, metadata: dict):
        """Handle algorithm completion."""
        self.is_running = False
        self.is_paused = False
        self.algorithm_generator = None
        
        if success and path:
            # Mark path
            self.grid.mark_path(path)
            
            # Update status
            path_length = metadata.get("path_cost", len(path) - 1)
            nodes_expanded = metadata.get("nodes_expanded", 0)
            elapsed = metadata.get("elapsed_time", 0)
            
            self.status_var.set(
                f"Path found! Length: {path_length}, "
                f"Nodes expanded: {nodes_expanded}, "
                f"Time: {elapsed:.3f}s"
            )
        else:
            # No path found
            self.status_var.set("No path found!")
            messagebox.showinfo("Result", "No path exists between start and end positions.")
    
    def _pause_algorithm(self):
        """Pause algorithm execution."""
        if self.is_running and not self.is_paused:
            self.is_paused = True
            if self.animation_id:
                self.root.after_cancel(self.animation_id)
                self.animation_id = None
            self.status_var.set("Paused")
    
    def _step_algorithm(self):
        """Step algorithm one iteration."""
        if not self.is_running:
            # Start if not running
            self._start_algorithm()
            self.is_paused = True
            return
        
        if not self.is_paused:
            return
        
        # Single step
        try:
            result = next(self.algorithm_generator)
            
            if isinstance(result, tuple) and len(result) == 3:
                success, path, metadata = result
                self._handle_algorithm_complete(success, path, metadata)
                return
            
            self.root.update_idletasks()
        except StopIteration:
            self._handle_algorithm_complete(False, None, {})
        except Exception as e:
            messagebox.showerror("Error", f"Algorithm error: {str(e)}")
            self._stop_algorithm()
    
    def _stop_algorithm(self):
        """Stop algorithm execution."""
        self.is_running = False
        self.is_paused = False
        
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
        
        self.algorithm_generator = None
        
        # Reset grid visualization (keep walls, start, end)
        self.grid.reset(keep_start_end=True)
        self.status_var.set("Stopped")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = AlgorithmVisualizer(root)
    
    # Ensure window is displayed and updated before grid initialization
    root.update()
    root.update_idletasks()
    
    root.mainloop()


if __name__ == "__main__":
    main()

