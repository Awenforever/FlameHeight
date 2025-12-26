# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 17:30
# @Author  : Jinhong Wu
# @File    : flame_height_v4.py
# @Project : Develop
# Copyright (c) 2025 Jinhong Wu. All rights reserved.
from dotenv import load_dotenv; load_dotenv()
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
import csv
import sys
from utils import console, progress
from shapely.geometry import Polygon, LineString

# Constants
FLAME_THRESHOLD = 50
LOWER_FLAME = [1, 46, 75]      # HSV Lower Bound for Flame Colors
UPPER_FLAME = [176, 255, 255]    # HSV Upper Bound
DEFAULT_REF_DISTANCE_CM = 1.0
PLOT_HEIGHT_RATIO = 0.3


class VideoAnalyzer:
    def __init__(self, input_paths, output_dir=None):
        # Load .env file from the current working directory or executable directory

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

        # Matplotlib Figure
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.canvas = None

        # Initialize settings with .env values if available, else defaults
        self.flame_threshold = int(os.getenv("FLAME_THRESHOLD", FLAME_THRESHOLD))
        self.ref_distance_cm = float(os.getenv("REF_DISTANCE_CM", DEFAULT_REF_DISTANCE_CM))

        # Parse HSV from .env (expects comma separated string)
        self.lower_flame = self._parse_env_hsv("LOWER_FLAME", LOWER_FLAME)
        self.upper_flame = self._parse_env_hsv("UPPER_FLAME", UPPER_FLAME)

        self.bottom_percentile = 90

    @staticmethod
    def _parse_env_hsv(key, default_list):
        env_val = os.getenv(key)
        if env_val:
            try:
                return np.array([int(x.strip()) for x in env_val.split(',')])
            except ValueError:
                return np.array(default_list)
        return np.array(default_list)

    def _configure_settings(self):
        """Modified: Loads from .env. If .env keys are missing, falls back to console."""

        # Check if at least one key exists in .env to decide whether to skip console input
        env_keys = ["FLAME_THRESHOLD", "REF_DISTANCE_CM", "LOWER_FLAME", "UPPER_FLAME"]
        using_env = any(os.getenv(k) is not None for k in env_keys)

        if using_env:
            console.rule("[bold green]Configuration Loaded from .env")
            console.print(f"   [blue]Threshold:[/blue] {self.flame_threshold}")
            console.print(f"   [blue]Ref Distance:[/blue] {self.ref_distance_cm} cm")
            console.print(f"   [blue]HSV Lower:[/blue] {self.lower_flame.tolist()}")
            console.print(f"   [blue]HSV Upper:[/blue] {self.upper_flame.tolist()}\n")
            return  # Skip manual input

        # Fallback to your original interactive console logic if no .env is found
        console.rule("[bold cyan]Configuration Settings (No .env detected)")
        console.print("[italic]Press [bold]Enter[/bold] to keep the default value.[/italic]\n")

        # 1. Flame Threshold
        val = console.input(f"   Flame Threshold ({self.flame_threshold}): ")
        if val.strip(): self.flame_threshold = int(val)

        # 2. Reference Distance
        val = console.input(f"   Ref Distance cm ({self.ref_distance_cm}): ")
        if val.strip(): self.ref_distance_cm = float(val)

        # 3 & 4 HSV Bounds (Your existing logic)
        def parse_hsv_manual(prompt, default_val):
            value = console.input(f"   {prompt} ({default_val.tolist()}): ")
            if value.strip():
                return np.array([int(x.strip()) for x in value.split(',')])
            return default_val

        self.lower_flame = parse_hsv_manual("Lower HSV Bound", self.lower_flame)
        self.upper_flame = parse_hsv_manual("Upper HSV Bound", self.upper_flame)
        console.print("\n[green]Settings updated successfully![/green]\n")

    def get_frame_generator(self):
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

    def get_total_frames(self):
        """Calculates the total frame count across all input videos."""
        total = 0
        for video_path in self.input_paths:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                # efficient way to get count without reading frames
                total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
        return total

    def setup_calibration(self):
        """Interactive setup with UNDO support for points and ROI."""
        gen = self.get_frame_generator()
        first_frame = next(gen)
        h, w = first_frame.shape[:2]

        self._calibration_step(first_frame)
        self._roi_selection_step(first_frame)
        self._nozzle_position_step(first_frame)

        cv2.destroyAllWindows()
        return first_frame.shape

    def _calibration_step(self, first_frame):
        """Step 1: Calibration Points with Undo."""
        h, w = first_frame.shape[:2]
        calib_points = []
        lines_px = []

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", min(w, 1280), min(h, 720))

        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                calib_points.append((x, y))
                if len(calib_points) % 2 == 0:
                    dist = np.linalg.norm(np.array(calib_points[-2]) - np.array(calib_points[-1]))
                    lines_px.append(dist)

        cv2.setMouseCallback("Calibration", click_callback)
        console.print("[yellow]STEP 1: Calibration.[/yellow]")
        console.print("  > [bold]Click start/end[/bold] of 1cm reference.")
        console.print("  > Press [bold]'z'[/bold] to undo last point/line.")
        console.print("  > Press [bold]'c'[/bold] to confirm and move to ROI.")

        while True:
            display_img = first_frame.copy()
            self._draw_calibration_points(display_img, calib_points)
            cv2.imshow("Calibration", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if len(lines_px) > 0:
                    break
                console.print("[red]Error: Define at least one 1cm line before confirming![/red]")
            elif key == ord('z'):
                self._undo_calibration_point(calib_points, lines_px)

        self.scale_ratio = np.divide(np.mean(lines_px), self.ref_distance_cm)

    @staticmethod
    def _draw_calibration_points(display_img, calib_points):
        """Draw points and lines on calibration image."""
        for i, pt in enumerate(calib_points):
            cv2.circle(display_img, pt, 3, (0, 0, 255), -1)
            if (i + 1) % 2 == 0:
                cv2.line(display_img, calib_points[i - 1], calib_points[i], (0, 255, 0), 2)

    @staticmethod
    def _undo_calibration_point(calib_points, lines_px):
        """Remove last calibration point and associated line."""
        if calib_points:
            if len(calib_points) % 2 == 0:
                lines_px.pop()
            calib_points.pop()
            console.print("[cyan]Last point removed.[/cyan]")

    def _roi_selection_step(self, first_frame):
        """Step 2: ROI Selection with Confirmation."""
        console.print("\n[yellow]STEP 2: Select ROI (Flame Area).[/yellow]")
        console.print("  > Draw box, then press [bold]ENTER/SPACE[/bold] to preview.")
        console.print("  > Press [bold]'SPACE'[/bold] to confirm ROI, or [bold]re-draw[/bold] if unhappy.")

        while True:
            self.roi = cv2.selectROI("Calibration", first_frame, showCrosshair=True)
            if self._confirm_roi(first_frame):
                break

    def _confirm_roi(self, first_frame):
        """Show ROI preview and request confirmation."""
        rx, ry, rw, rh = self.roi
        preview = first_frame.copy()
        cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)
        cv2.putText(preview, "Press 'c' to confirm this ROI", (rx, ry - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Calibration", preview)
        return (cv2.waitKey(0) & 0xFF) == ord('c')

    def _nozzle_position_step(self, first_frame):
        """Step 3: Nozzle Position Setup."""
        rx, ry, rw, rh = self.roi
        roi_img = first_frame[ry:ry + rh, rx:rx + rw].copy()
        nozzle_local_y = rh - 1

        def nozzle_callback(event, x, y, flags, param):
            nonlocal nozzle_local_y
            if event == cv2.EVENT_LBUTTONDOWN:
                nozzle_local_y = y

        cv2.namedWindow("Set Nozzle Top")
        cv2.setMouseCallback("Set Nozzle Top", nozzle_callback)
        console.print("\n[yellow]STEP 3: Define Nozzle Top.[/yellow]")
        console.print("  > [bold]Click[/bold] the nozzle edge. Press [bold]'c'[/bold] to confirm.\n")

        while True:
            temp = roi_img.copy()
            cv2.line(temp, (0, nozzle_local_y), (rw, nozzle_local_y), (0, 0, 255), 2)
            cv2.imshow("Set Nozzle Top", temp)
            if (cv2.waitKey(1) & 0xFF) == ord('c'):
                break

        self.nozzle_y = nozzle_local_y

    @staticmethod
    def find_flame_root_points(contour, percentile=10):
        """
        contour: OpenCV contour, shape (N,1,2)
        返回两个点 (x1,y1), (x2,y2)
        """
        contour = contour.reshape(-1, 2)
        ys = contour[:, 1]
        y_thresh = np.percentile(ys, percentile)

        bottom_indices = np.where(ys >= y_thresh)[0]
        if len(bottom_indices) < 2:
            return None

        best_pair = None
        best_score = -1e18

        for i_idx in range(len(bottom_indices)):
            for j_idx in range(i_idx + 1, len(bottom_indices)):
                i = bottom_indices[i_idx]
                j = bottom_indices[j_idx]

                p1 = contour[i]
                p2 = contour[j]

                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])
                dist = np.hypot(dx, dy)

                # 取轮廓段
                if i <= j:
                    seg = contour[i:j + 1]
                else:
                    seg = contour[j:i + 1]

                # 使用 shapely 计算面积
                # 多边形顶点顺序：p1 -> seg -> p2
                poly_pts = [tuple(p1)] + [tuple(pt) for pt in seg] + [tuple(p2)]
                polygon = Polygon(poly_pts)
                area = polygon.area if polygon.is_valid else 1e9

                # 评分函数
                score = dx + 0.5 * dist - 2.0 * area

                if score > best_score:
                    best_score = score
                    best_pair = (tuple(p1), tuple(p2))

        return best_pair

    def analyze_frame(self, frame, frame_idx):
        rx, ry, rw, rh = self.roi
        roi_frame = frame[ry:ry + rh, rx:rx + rw]

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        _, v_mask = cv2.threshold(hsv[:, :, 2], self.flame_threshold, 255, cv2.THRESH_BINARY)
        color_mask = cv2.inRange(hsv, self.lower_flame, self.upper_flame)
        mask = cv2.bitwise_and(v_mask, color_mask)

        mask[self.nozzle_y:, :] = 0

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dist_px, dist_cm, dist_std = 0.0, 0.0, 0.0

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 15:

                root_pair = self.find_flame_root_points(c, percentile=self.bottom_percentile)

                if root_pair is not None:
                    (x1, y1), (x2, y2) = root_pair
                    xl, xr = sorted([x1, x2])

                    contour_flat = c.reshape(-1, 2)

                    # 取轮廓中 x ∈ [xl, xr] 的点
                    region_pts = contour_flat[(contour_flat[:, 0] >= xl) & (contour_flat[:, 0] <= xr)]

                    if len(region_pts) > 0:
                        ys = region_pts[:, 1]
                        min_y = float(np.max(ys))  # 火焰底部真实轮廓点

                        avg_bottom_y = min_y

                        dist_px = max(0.0, self.nozzle_y - avg_bottom_y)
                        dist_cm = dist_px / self.scale_ratio
                        dist_std = float(np.std(ys) / self.scale_ratio)

                        # 绘制轮廓与底部线
                        cv2.drawContours(roi_frame, [c], -1, (0, 255, 0), 1)
                        cv2.line(roi_frame,
                                 (xl, int(avg_bottom_y)),
                                 (xr, int(avg_bottom_y)),
                                 (255, 0, 0), 1)

        cv2.line(roi_frame, (0, self.nozzle_y), (rw, self.nozzle_y), (0, 0, 255), 1)
        frame[ry:ry + rh, rx:rx + rw] = roi_frame

        self.frame_indices.append(frame_idx)
        self.history_dist_px.append(dist_px)
        self.history_dist_cm.append(dist_cm)
        self.history_dist_std_cm.append(dist_std)
        return frame, dist_cm

    def init_plot(self, video_w):
        """Initialize figure with minimal margins."""
        dpi = 100
        raw_h = int(video_w * PLOT_HEIGHT_RATIO * 0.5)
        self.plot_h = raw_h if raw_h % 2 == 0 else raw_h + 1

        w_inch = video_w / dpi
        h_inch = self.plot_h / dpi

        self.fig, self.ax1 = plt.subplots(figsize=(w_inch, h_inch), dpi=dpi)
        self.ax2 = self.ax1.twinx()

        # Narrowed margins:
        # Left reduced from 0.08 to 0.05
        # Right increased from 0.88 to 0.94 (to fill the width while keeping label space)
        # Bottom/Top tightened
        self.fig.subplots_adjust(left=0.05, right=0.94, top=0.95, bottom=0.20)

        self.canvas = FigureCanvas(self.fig)

    def create_plot_image(self, target_width, target_height):
        """Generates the plot with tight x-axis limits and projection lines."""
        self.ax1.clear()
        self.ax2.clear()

        FONT_LABEL = 16
        FONT_TICK = 14
        FONT_HUD = 16
        FONT_LEGEND = 10

        x_data = np.array(self.frame_indices)
        if len(x_data) == 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        y_mm = np.array(self.history_dist_cm) * 10
        y_std_mm = np.array(self.history_dist_std_cm) * 10

        color_line = '#1f77b4'
        self.ax1.plot(x_data, y_mm, color=color_line, linewidth=2.0, label='Height')
        self.ax1.fill_between(x_data, y_mm - y_std_mm, y_mm + y_std_mm,
                              color=color_line, alpha=0.3, label='Std Dev')

        # Formatting
        current_max_x = max(100, x_data[-1])
        self.ax1.set_xlim(left=0, right=current_max_x)
        self.ax1.set_xlabel("Frame Index", fontsize=FONT_LABEL, fontweight='bold')
        self.ax1.tick_params(axis='x', labelsize=FONT_TICK)

        # Left Y-Axis
        self.ax1.set_ylabel("Height (mm)", color=color_line, fontsize=FONT_LABEL, fontweight='bold')
        self.ax1.tick_params(axis='y', labelcolor=color_line, labelsize=FONT_TICK)
        self.ax1.grid(True, linestyle='--', alpha=0.6, linewidth=0.7)

        # Right Y-Axis (Pixels)
        self.ax2.set_ylabel("Pixels (px)", color='gray', fontsize=FONT_LABEL, fontweight='bold', labelpad=12,
                            rotation=270)
        self.ax2.yaxis.set_label_position("right")
        self.ax2.tick_params(axis='y', labelcolor='gray', labelsize=FONT_TICK, right=True, labelright=True)

        # Sync Y-Axes
        y1_min, y1_max = self.ax1.get_ylim()
        if y1_max == y1_min: y1_max += 1.0
        y2_min = (y1_min / 10.0) * self.scale_ratio
        y2_max = (y1_max / 10.0) * self.scale_ratio
        self.ax2.set_ylim(y2_min, y2_max)

        # --- ADD PROJECTION LINES TO CURRENT POINT ---
        curr_x = x_data[-1]
        curr_mm = y_mm[-1]
        curr_px = self.history_dist_px[-1]
        curr_std_mm = y_std_mm[-1]

        # Vertical line to X-axis
        self.ax1.axvline(x=curr_x, color='red', linestyle='--', linewidth=1, alpha=0.7)
        # Horizontal line to Y-axis
        self.ax1.axhline(y=curr_mm, color='red', linestyle='--', linewidth=1, alpha=0.7)

        # Highlight Current Point
        self.ax1.scatter([curr_x], [curr_mm], color='red', s=80, zorder=10, edgecolors='white')

        # Visuals
        self.ax1.legend(loc='upper right', fontsize=FONT_LEGEND, framealpha=0.9)

        label_text = f"Height: {curr_mm:.2f}±{curr_std_mm:.2f} mm\nPixels: {curr_px:.1f} px"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        self.ax1.text(0.02, 0.96, label_text, transform=self.ax1.transAxes,
                      fontsize=FONT_HUD, verticalalignment='top', bbox=props, zorder=15)

        self.canvas.draw()
        buf = np.array(self.canvas.renderer.buffer_rgba())
        img_plot = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

        if img_plot.shape[0] != target_height or img_plot.shape[1] != target_width:
            img_plot = cv2.resize(img_plot, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return img_plot

    def run(self):
        self._configure_settings()
        vh, vw, _ = self.setup_calibration()
        self.init_plot(vw)
        total_h = vh + self.plot_h
        out = cv2.VideoWriter(str(self.video_out_path), cv2.VideoWriter_fourcc(*'mp4v'), 50.0, (vw, total_h))

        with progress:
            task = progress.add_task("[cyan]Processing...", total=self.get_total_frames())
            for i, frame in enumerate(self.get_frame_generator()):
                processed, h_cm = self.analyze_frame(frame, i)
                plot = self.create_plot_image(vw, self.plot_h)
                out.write(np.vstack((processed, plot)))
                progress.update(task, description=f"Frame {i} | H: {h_cm * 10:.2f}mm", advance=1)

        out.release()
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Dist_Px', 'Dist_CM', 'Dist_Std_CM'])
            writer.writerows(
                zip(self.frame_indices, self.history_dist_px, self.history_dist_cm, self.history_dist_std_cm))


if __name__ == "__main__":
    if __name__ == "__main__":
        # 1. Check if paths were passed via command line (drag and drop)
        input_paths = sys.argv[1:]

        # 2. If no paths, pop up a file selection dialog
        if not input_paths:
            root = tk.Tk()
            root.withdraw()  # Hide the main tiny Tkinter window
            root.attributes('-topmost', True)  # Bring the dialog to the front

            # Define supported video extensions for cv2.VideoCapture
            video_types = [
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.mts"),
                ("All files", "*.*")
            ]

            selected_files = filedialog.askopenfilenames(
                title="Select Flame Videos to Analyze",
                filetypes=video_types
            )

            if selected_files:
                input_paths = list(selected_files)
            else:
                print("No files selected. Exiting.")
                sys.exit()

        # 3. Run the analyzer
        VideoAnalyzer(input_paths).run()