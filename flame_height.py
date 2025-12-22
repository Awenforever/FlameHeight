# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 16:37
# @Author  : Jinhong Wu
# @File    : flame_height.py
# @Project : Develop
# Copyright (c) 2025 Jinhong Wu. All rights reserved.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
from rich.progress import Progress
from rich.console import Console
import csv
import argparse
import sys

# Constants
DEFAULT_REF_DISTANCE_CM = 2.0  # The real distance of drawn calibration lines
FLAME_THRESHOLD = 50           # Brightness threshold for flame detection (adjustable)

console = Console()

class VideoAnalyzer:
    def __init__(self, input_paths, output_dir=None):
        self.input_paths = [Path(p) for p in input_paths]
        self.input_paths.sort()  # Ensure sequential order

        # Output Setup
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.input_paths[0].parent

        self.output_stem = self.input_paths[0].stem + "_analyzed"
        self.csv_path = self.output_dir / f"{self.output_stem}.csv"
        self.video_out_path = self.output_dir / f"{self.output_stem}.mp4"

        # Analysis State
        self.roi = None  # (x, y, w, h)
        self.scale_ratio = None  # pixels per cm
        self.nozzle_y = None # The y-coordinate of the nozzle top (relative to ROI or Frame)

        # Data History
        self.history_dist_px = []
        self.history_dist_cm = []
        self.history_dist_std_cm = []
        self.frame_indices = []

        # Matplotlib Figure (Reusable)
        self.fig, self.ax1 = plt.subplots(figsize=(5, 3), dpi=100)
        self.ax2 = self.ax1.twinx()
        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout()

    def get_frame_generator(self):
        """Yields frames sequentially from all input videos."""
        for video_path in self.input_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                console.print(f"[red]Error opening {video_path}[/red]")
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

    def setup_calibration(self):
        """Interactive setup for ROI, Scale, and Nozzle position."""
        # Get first frame
        gen = self.get_frame_generator()
        first_frame = next(gen)
        display_img = first_frame.copy()

        # --- Step 1: Calibration Lines ---
        lines = []
        drawing = False
        start_point = (-1, -1)

        def line_mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, display_img
            temp_img = display_img.copy()

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.line(temp_img, start_point, (x, y), (0, 255, 255), 2)
                    cv2.imshow("Calibration", temp_img)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
                cv2.line(display_img, start_point, end_point, (0, 255, 0), 2)

                # Calculate Euclidean distance
                dist_px = np.linalg.norm(np.array(start_point) - np.array(end_point))
                lines.append(dist_px)
                cv2.imshow("Calibration", display_img)
                console.print(f"  Line added: {dist_px:.2f} px")

        console.print("[yellow]STEP 1: Draw calibration lines.[/yellow]")
        console.print(f"  > Draw lines representing {DEFAULT_REF_DISTANCE_CM}cm.")
        console.print("  > Press 'c' when done to confirm.")

        cv2.imshow("Calibration", display_img)
        cv2.setMouseCallback("Calibration", line_mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if not lines:
                    console.print("[red]Draw at least one line![/red]")
                    continue
                break
        
        # Calculate Scale
        avg_px = np.mean(lines)
        self.scale_ratio = avg_px / DEFAULT_REF_DISTANCE_CM
        console.print(f"[green]Scale calibrated: {self.scale_ratio:.2f} px/cm[/green]")

        # --- Step 2: ROI Selection ---
        console.print("\n[yellow]STEP 2: Select ROI.[/yellow]")
        console.print("  > Draw a rectangle enclosing the flame and nozzle.")
        console.print("  > Press ENTER/SPACE to confirm.")

        # Note: SelectROI is blocking
        self.roi = cv2.selectROI("Calibration", display_img, showCrosshair=True)
        # roi format: (x, y, w, h)

        # --- Step 3: Nozzle Position ---
        # We work on the cropped ROI now to ensure alignment
        rx, ry, rw, rh = self.roi
        roi_img = display_img[ry:ry+rh, rx:rx+rw].copy()
        nozzle_local_y = rh - 1 # Default to bottom

        def nozzle_callback(event, x, y, flags, param):
            nonlocal nozzle_local_y, roi_img
            temp_roi = roi_img.copy()
            if event == cv2.EVENT_LBUTTONDOWN:
                nozzle_local_y = y
                cv2.line(temp_roi, (0, y), (rw, y), (0, 0, 255), 2)
                cv2.imshow("Set Nozzle Top", temp_roi)

        console.print("\n[yellow]STEP 3: Define Nozzle Top.[/yellow]")
        console.print("  > Click the top edge of the nozzle inside the ROI window.")
        console.print("  > Press 'c' to confirm.")

        cv2.imshow("Set Nozzle Top", roi_img)
        cv2.setMouseCallback("Set Nozzle Top", nozzle_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                break

        self.nozzle_y = nozzle_local_y
        cv2.destroyAllWindows()
        return first_frame.shape

    def analyze_frame(self, frame, frame_idx):
        rx, ry, rw, rh = self.roi

        # Crop to ROI
        roi_frame = frame[ry:ry+rh, rx:rx+rw]

        # 1. Image Preprocessing (Flame Detection)
        # Convert to gray. Flames are often bright.
        # Alternatively, use Blue channel if flame is blue.
        # Let's use Value channel from HSV or just simple Grayscale thresholding.
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Threshold to get flame mask
        _, mask = cv2.threshold(gray, FLAME_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dist_px = 0.0
        dist_cm = 0.0
        dist_std = 0.0
        flame_found = False

        if contours:
            # Find largest contour (main flame)
            c = max(contours, key=cv2.contourArea)

            # Filter noise: ignore very small contours
            if cv2.contourArea(c) > 50:
                flame_found = True

                # --- Compute Distance ---
                # We need the bottom profile of the flame.
                # c is shape (N, 1, 2) -> (x, y)
                points = c[:, 0, :]

                # We need all (x, y) pixels *inside* or on the edge of the contour.
                # A robust way: Iterate columns in the bounding box of the contour,
                # find the lowest y pixel that is part of the mask.
                x_start, y_start, w_c, h_c = cv2.boundingRect(c)

                bottom_y_values = []

                # Scan columns within the flame's bounding box
                for x_col in range(x_start, x_start + w_c):
                    # Extract column from mask
                    col_mask = mask[:, x_col]
                    # Find indices where mask is 255
                    foreground_indices = np.where(col_mask == 255)[0]
                    if len(foreground_indices) > 0:
                        # Max Y is the bottom-most pixel
                        bottom_y_values.append(foreground_indices[-1])

                if bottom_y_values:
                    # Calculate stats
                    avg_bottom_y = np.mean(bottom_y_values)
                    std_bottom_y = np.std(bottom_y_values)

                    # Distance = Nozzle Y - Average Flame Bottom Y
                    # (Y increases downwards)
                    dist_px = max(0, self.nozzle_y - avg_bottom_y)
                    dist_cm = dist_px / self.scale_ratio

                    # Std in real units
                    dist_std = std_bottom_y / self.scale_ratio

                # --- Draw Outline on ROI ---
                cv2.drawContours(roi_frame, [c], -1, (0, 255, 0), 1)

                # Draw "Average Bottom" line for visualization
                y_vis = int(avg_bottom_y)
                cv2.line(roi_frame, (x_start, y_vis), (x_start+w_c, y_vis), (255, 0, 0), 1)

        # Draw Nozzle Line (Stationary)
        cv2.line(roi_frame, (0, self.nozzle_y), (rw, self.nozzle_y), (0, 0, 255), 1)

        # Put processed ROI back into frame
        frame[ry:ry+rh, rx:rx+rw] = roi_frame

        # Store Data
        self.frame_indices.append(frame_idx)
        self.history_dist_px.append(dist_px)
        self.history_dist_cm.append(dist_cm)
        self.history_dist_std_cm.append(dist_std)

        return frame, dist_px, dist_cm

    def create_plot_overlay(self, current_frame_idx):
        """Generates the matplotlib chart as an image array."""
        self.ax1.clear()
        self.ax2.clear()

        x_data = self.frame_indices
        y_px = np.array(self.history_dist_px)
        y_cm = np.array(self.history_dist_cm)
        y_err = np.array(self.history_dist_std_cm)

        # Plot Pixels (Left Axis)
        color1 = 'tab:blue'
        self.ax1.set_xlabel('Frame')
        self.ax1.set_ylabel('Distance (px)', color=color1)
        self.ax1.plot(x_data, y_px, color=color1, alpha=0.6, label='Pixels')
        self.ax1.tick_params(axis='y', labelcolor=color1)

        # Plot Real Distance (Right Axis) with Error Band
        color2 = 'tab:orange'
        self.ax2.set_ylabel('Distance (cm)', color=color2)
        self.ax2.plot(x_data, y_cm, color=color2, linewidth=2, label='Real Dist')

        # Error Band
        self.ax2.fill_between(x_data, y_cm - y_err, y_cm + y_err, color=color2, alpha=0.2)

        # Highlight current point
        if len(x_data) > 0:
            self.ax2.scatter([x_data[-1]], [y_cm[-1]], color='red', s=50, zorder=5)
            # Text annotation
            curr_cm = y_cm[-1]
            self.ax2.text(x_data[-1], curr_cm, f"{curr_cm:.2f}cm", fontsize=8, verticalalignment='bottom')

        self.ax2.tick_params(axis='y', labelcolor=color2)

        # Render to canvas
        self.canvas.draw()

        # Convert to numpy array
        # Get RGBA buffer
        buf = np.array(self.canvas.renderer.buffer_rgba())
        # Convert RGBA to BGR for OpenCV
        img_plot = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

        return img_plot

    def run(self):
        # 1. Setup
        frame_shape = self.setup_calibration()
        h, w, _ = frame_shape

        # Initialize Video Writer
        # H.264 encoding via 'mp4v' or 'avc1'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.video_out_path), fourcc, 30.0, (w, h))

        console.print(f"[bold green]Starting Analysis...[/bold green]")
        console.print(f"Output Video: {self.video_out_path}")
        console.print(f"Output CSV: {self.csv_path}")

        frame_gen = self.get_frame_generator()
        frame_idx = 0

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing Frames...", total=None) # Unknown total length

            for frame in frame_gen:
                if frame is None: break

                # Analyze
                processed_frame, dist_px, dist_cm = self.analyze_frame(frame, frame_idx)

                # Create Plot Overlay
                # Resize plot to fit in top-left or overlay cleanly
                plot_img = self.create_plot_overlay(frame_idx)

                # Overlay logic: Place at top-left
                ph, pw, _ = plot_img.shape
                # Ensure plot fits
                if ph < h and pw < w:
                    processed_frame[0:ph, 0:pw] = plot_img

                # Write to video
                out.write(processed_frame)

                progress.update(task, advance=1, description=f"Frame {frame_idx} | Dist: {dist_cm:.2f}cm")
                frame_idx += 1

        out.release()

        # Save CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Distance_Px', 'Distance_CM', 'Distance_Std_CM'])
            for i in range(len(self.frame_indices)):
                writer.writerow([
                    self.frame_indices[i],
                    self.history_dist_px[i],
                    self.history_dist_cm[i],
                    self.history_dist_std_cm[i]
                ])

        console.print("[bold green]Processing Complete![/bold green]")

if __name__ == "__main__":
    # Example usage: python flame_measure.py video1.mts video2.mts
    if len(sys.argv) < 2:
        console.print("[red]Usage: python flame_measure.py <video1.mts> [video2.mts ...][/red]")
    else:
        # Pass all arguments except script name
        analyzer = VideoAnalyzer(sys.argv[1:])
        analyzer.run()
