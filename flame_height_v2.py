# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 16:37
# @Author  : Jinhong Wu
# @File    : flame_height_v2.py
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
DEFAULT_REF_DISTANCE_CM = 1.0  # The real distance of drawn calibration lines
FLAME_THRESHOLD = 75  # Brightness threshold
PLOT_HEIGHT_RATIO = 0.3  # Plot height relative to video height (e.g., 30%)

console = Console()


class VideoAnalyzer:
    def __init__(self, input_paths, output_dir=None):
        self.input_paths = [Path(p) for p in input_paths]
        self.input_paths.sort()

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.input_paths[0].parent

        self.output_stem = self.input_paths[0].stem + "_analyzed"
        self.csv_path = self.output_dir / f"{self.output_stem}.csv"
        self.video_out_path = self.output_dir / f"{self.output_stem}.mp4"

        # Analysis State
        self.roi = None
        self.scale_ratio = None
        self.nozzle_y = None

        # Data History
        self.history_dist_px = []
        self.history_dist_cm = []
        self.history_dist_std_cm = []
        self.frame_indices = []

        # Matplotlib Figure (Initialized later based on video size)
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.canvas = None

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
        """Interactive setup for Scale, ROI, and Nozzle position."""
        gen = self.get_frame_generator()
        first_frame = next(gen)
        display_img = first_frame.copy()
        h, w = display_img.shape[:2]

        # --- Step 1: Calibration Lines (Point-to-Point) ---
        calib_points = []  # Stores tuples (x, y)
        lines_px = []

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", min(w, 1280), min(h, 720))

        def click_callback(event, x, y, flags, param):
            nonlocal display_img
            if event == cv2.EVENT_LBUTTONDOWN:
                calib_points.append((x, y))
                # Visualize point
                cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)

                # If we have a pair, draw line
                if len(calib_points) % 2 == 0:
                    pt1 = calib_points[-2]
                    pt2 = calib_points[-1]
                    cv2.line(display_img, pt1, pt2, (0, 255, 0), 2)
                    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
                    lines_px.append(dist)
                    console.print(f"  Line added: {dist:.2f} px")

                cv2.imshow("Calibration", display_img)

        console.print("[yellow]STEP 1: Calibration.[/yellow]")
        console.print("  > Click Point A then Point B to define a line.")
        console.print(f"  > This line represents {DEFAULT_REF_DISTANCE_CM}cm.")
        console.print("  > Press 'z' to undo last line.")
        console.print("  > Press 'c' to confirm.")

        cv2.imshow("Calibration", display_img)
        cv2.setMouseCallback("Calibration", click_callback)

        temp_clean_img = display_img.copy()  # Backup for undo

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if not lines_px:
                    console.print("[red]Define at least one line![/red]")
                    continue
                break
            elif key == ord('z'):
                # Undo logic
                if len(calib_points) >= 2:
                    calib_points.pop()
                    calib_points.pop()
                    lines_px.pop()
                    console.print("  [yellow]Last line removed.[/yellow]")

                    # Redraw
                    display_img = temp_clean_img.copy()
                    # Redraw remaining lines
                    for i in range(0, len(calib_points), 2):
                        pt1 = calib_points[i]
                        pt2 = calib_points[i + 1]
                        cv2.circle(display_img, pt1, 3, (0, 0, 255), -1)
                        cv2.circle(display_img, pt2, 3, (0, 0, 255), -1)
                        cv2.line(display_img, pt1, pt2, (0, 255, 0), 2)
                    cv2.imshow("Calibration", display_img)

        # Calculate Scale
        avg_px = np.mean(lines_px)
        self.scale_ratio = avg_px / DEFAULT_REF_DISTANCE_CM
        console.print(f"[green]Scale calibrated: {self.scale_ratio:.2f} px/cm[/green]")

        # --- Step 2: ROI Selection ---
        console.print("\n[yellow]STEP 2: Select ROI.[/yellow]")
        console.print("  > Draw a rectangle enclosing the flame.")
        console.print("  > Press SPACE/ENTER to confirm.")

        # Reset callback for ROI selection
        cv2.setMouseCallback("Calibration", lambda *args: None)
        self.roi = cv2.selectROI("Calibration", display_img, showCrosshair=True)

        # --- Step 3: Nozzle Position ---
        rx, ry, rw, rh = self.roi
        roi_img = display_img[ry:ry + rh, rx:rx + rw].copy()
        nozzle_local_y = rh - 1

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
        roi_frame = frame[ry:ry + rh, rx:rx + rw]

        # 1. Improved Preprocessing
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur to remove high freq noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold
        _, mask = cv2.threshold(blurred, FLAME_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Morphological Operations (Crucial for background noise)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)  # Remove small white noise
        mask = cv2.dilate(mask, kernel, iterations=2)  # Fill gaps in flame

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dist_px = 0.0
        dist_cm = 0.0
        dist_std = 0.0

        if contours:
            c = max(contours, key=cv2.contourArea)

            # Area filter
            if cv2.contourArea(c) > 50:
                # Find bottom points via column scanning
                x_start, y_start, w_c, h_c = cv2.boundingRect(c)
                bottom_y_values = []

                for x_col in range(x_start, x_start + w_c):
                    col_mask = mask[:, x_col]
                    foreground_indices = np.where(col_mask == 255)[0]
                    if len(foreground_indices) > 0:
                        bottom_y_values.append(foreground_indices[-1])

                if bottom_y_values:
                    avg_bottom_y = np.mean(bottom_y_values)
                    std_bottom_y = np.std(bottom_y_values)

                    dist_px = max(0, self.nozzle_y - avg_bottom_y)
                    dist_cm = dist_px / self.scale_ratio
                    dist_std = std_bottom_y / self.scale_ratio

                # Draw Visuals
                cv2.drawContours(roi_frame, [c], -1, (0, 255, 0), 1)
                y_vis = int(avg_bottom_y) if bottom_y_values else 0
                cv2.line(roi_frame, (x_start, y_vis), (x_start + w_c, y_vis), (255, 0, 0), 1)

        cv2.line(roi_frame, (0, self.nozzle_y), (rw, self.nozzle_y), (0, 0, 255), 1)
        frame[ry:ry + rh, rx:rx + rw] = roi_frame

        self.frame_indices.append(frame_idx)
        self.history_dist_px.append(dist_px)
        self.history_dist_cm.append(dist_cm)
        self.history_dist_std_cm.append(dist_std)

        return frame, dist_cm

    def init_plot(self, video_w):
        """Initialize figure with larger fonts and tighter layout."""
        dpi = 100
        # 1. Calculate Plot Height (approx 30% of video width)
        raw_h = int(video_w * PLOT_HEIGHT_RATIO * 0.5)

        # 2. ENFORCE EVEN NUMBER (Crucial for video codecs)
        self.plot_h = raw_h if raw_h % 2 == 0 else raw_h + 1

        w_inch = video_w / dpi
        h_inch = self.plot_h / dpi

        # Create figure
        self.fig, self.ax1 = plt.subplots(figsize=(w_inch, h_inch), dpi=dpi)
        self.ax2 = self.ax1.twinx()

        # Layout Adjustment
        # - Increased right margin (0.88 instead of 0.92) to accommodate the right y-label
        self.fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.22)

        self.canvas = FigureCanvas(self.fig)

    def create_plot_image(self, target_width, target_height):
        """Generates the plot with high-legibility fonts."""
        self.ax1.clear()
        self.ax2.clear()

        # --- Style Configuration ---
        FONT_LABEL = 16  # Axis Titles (e.g., "Height")
        FONT_TICK = 14  # Axis Numbers (e.g., "10, 20")
        FONT_HUD = 16  # The floating text box
        FONT_LEGEND = 10  # Legend text

        # --- Data Preparation ---
        x_data = np.array(self.frame_indices)
        if len(x_data) == 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        y_mm = np.array(self.history_dist_cm) * 10
        y_std_mm = np.array(self.history_dist_std_cm) * 10

        # --- Plotting ---
        color_line = '#1f77b4'
        self.ax1.plot(x_data, y_mm, color=color_line, linewidth=2.0, label='Height')
        self.ax1.fill_between(x_data, y_mm - y_std_mm, y_mm + y_std_mm,
                              color=color_line, alpha=0.3, label='Std Dev')

        # --- Formatting ---
        # X-Axis
        self.ax1.set_xlim(left=0, right=max(100, len(x_data)))
        self.ax1.set_xlabel("Frame Index", fontsize=FONT_LABEL, fontweight='bold')
        self.ax1.tick_params(axis='x', labelsize=FONT_TICK)

        # Left Y-Axis (mm)
        self.ax1.set_ylabel("Height (mm)", color=color_line, fontsize=FONT_LABEL, fontweight='bold')
        self.ax1.tick_params(axis='y', labelcolor=color_line, labelsize=FONT_TICK)
        self.ax1.grid(True, linestyle='--', alpha=0.6, linewidth=0.7)

        # Right Y-Axis (Pixels)
        # Explicitly set label position to the right and added padding for clarity
        self.ax2.set_ylabel("Pixels (px)", color='gray', fontsize=FONT_LABEL, fontweight='bold', labelpad=20, rotation=270)
        self.ax2.yaxis.set_label_position("right")
        self.ax2.tick_params(axis='y', labelcolor='gray', labelsize=FONT_TICK, right=True, labelright=True)

        # Sync Y-Axes
        y1_min, y1_max = self.ax1.get_ylim()
        # Avoid division by zero or flat lines
        if y1_max == y1_min: y1_max += 1.0

        y2_min = (y1_min / 10.0) * self.scale_ratio
        y2_max = (y1_max / 10.0) * self.scale_ratio
        self.ax2.set_ylim(y2_min, y2_max)

        # --- Visuals ---
        # Legend (Top Right)
        self.ax1.legend(loc='upper right', fontsize=FONT_LEGEND, framealpha=0.9)

        # Highlight Current Point
        curr_x = x_data[-1]
        curr_mm = y_mm[-1]
        curr_px = self.history_dist_px[-1]
        curr_std_mm = y_std_mm[-1]

        self.ax1.scatter([curr_x], [curr_mm], color='red', s=80, zorder=10, edgecolors='white')

        # HUD Label (Top Left)
        label_text = f"Height: {curr_mm:.2f} mm\nPixels: {curr_px:.1f} px\nStdDev: ±{curr_std_mm:.2f} mm"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        self.ax1.text(0.02, 0.96, label_text, transform=self.ax1.transAxes,
                      fontsize=FONT_HUD, verticalalignment='top', bbox=props, zorder=15)

        # --- Rendering ---
        self.canvas.draw()
        buf = np.array(self.canvas.renderer.buffer_rgba())
        img_plot = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

        # Strict Resize
        if img_plot.shape[0] != target_height or img_plot.shape[1] != target_width:
            img_plot = cv2.resize(img_plot, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return img_plot

    def run(self):
        # 1. Calibration
        frame_shape = self.setup_calibration()
        vh, vw, _ = frame_shape  # vh=Height, vw=Width

        # 2. Initialize Plot Size
        self.init_plot(vw)
        ph = self.plot_h  # This is now guaranteed to be even

        # Calculate Total Dimensions
        total_h = vh + ph
        total_w = vw

        # Ensure Total Height is even (just in case vh was odd)
        if total_h % 2 != 0:
            total_h += 1

        console.print(f"[cyan]Video Dims: {vw}x{vh}[/cyan]")
        console.print(f"[cyan]Plot Dims:  {vw}x{ph}[/cyan]")
        console.print(f"[cyan]Total Out:  {total_w}x{total_h}[/cyan]")

        # 3. Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(self.video_out_path), fourcc, 30.0, (total_w, total_h))

        if not out.isOpened():
            console.print("[yellow]Warning: 'avc1' codec failed. Falling back to 'mp4v'.[/yellow]")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(self.video_out_path), fourcc, 30.0, (total_w, total_h))

        if not out.isOpened():
            console.print(f"[bold red]CRITICAL ERROR: Could not open VideoWriter for {self.video_out_path}[/bold red]")
            return

        console.print(f"[bold green]\nStarting Analysis...[/bold green]")

        frame_gen = self.get_frame_generator()
        frame_idx = 0

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=None)

            for frame in frame_gen:
                if frame is None: break

                if frame.shape[0] != vh or frame.shape[1] != vw:
                    frame = cv2.resize(frame, (vw, vh))

                processed_frame, dist_cm = self.analyze_frame(frame, frame_idx)
                plot_img = self.create_plot_image(target_width=vw, target_height=ph)
                final_frame = np.vstack((processed_frame, plot_img))

                if final_frame.shape[:2] != (total_h, total_w):
                    final_frame = cv2.resize(final_frame, (total_w, total_h))

                final_frame = final_frame.astype(np.uint8)
                out.write(final_frame)
                progress.update(task, advance=1, description=f"Frame {frame_idx} | H: {dist_cm * 10:.2f}mm")
                frame_idx += 1

        out.release()

        # Save CSV
        try:
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
        except Exception as e:
            console.print(f"[red]Error saving CSV: {e}[/red]")

        console.print("[bold green]Processing Complete![/bold green]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python flame_measure.py <video_path>[/red]")
    else:
        analyzer = VideoAnalyzer(sys.argv[1:])
        analyzer.run()