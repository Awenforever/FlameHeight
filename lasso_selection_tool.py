# -*- coding: utf-8 -*-
# @Time    : 2025/12/20
# @Author  : Jinhong Wu
# @File    : lasso_selection_tool_v2.py
# @Project : Develop
# Copyright (c) 2025 Jinhong Wu. All rights reserved.

import cv2
import numpy as np
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog
from rich.progress import Progress
from rich.console import Console

console = Console()


class FullVideoLassoStats:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.points = []
        self.drawing = False
        self.current_mouse_pos = (0, 0)
        self.window_name = "Lasso Selection & Video Scrubbing"

    def mouse_callback(self, event, x, y, flags, param):
        """Handles drawing and hover tracking."""
        self.current_mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.points) > 2:
                self.points.append(self.points[0])

    def run_analysis(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            console.print("[red]Could not open video.[/red]")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Create Trackbar for scrubbing
        def on_trackbar(val):
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)

        cv2.createTrackbar("Frame", self.window_name, 0, total_frames - 1, on_trackbar)

        console.print("[bold yellow]Controls:[/bold yellow]")
        console.print("  > [bold]Trackbar[/bold]: Drag to scrub through video time.")
        console.print("  > [bold]Left-Click & Drag[/bold]: Draw the lasso.")
        console.print("  > [bold]'z'[/bold]: Undo last segment | [bold]'r'[/bold]: Reset lasso.")
        console.print("  > [bold]'c'[/bold]: Confirm selection and run full analysis.")

        while True:
            # Read frame at current trackbar position
            frame_idx = cv2.getTrackbarPos("Frame", self.window_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back if end of video
                continue

            display_img = frame.copy()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = frame.shape[:2]

            # --- Real-time HSV Hover (Pure White Text) ---
            mx, my = self.current_mouse_pos
            if 0 <= mx < w and 0 <= my < h:
                pixel_hsv = hsv_frame[my, mx]
                text = f"H:{pixel_hsv[0]} S:{pixel_hsv[1]} V:{pixel_hsv[2]}"
                # Pure white text (255, 255, 255)
                cv2.putText(display_img, text, (mx + 15, my + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # --- Draw Lasso ---
            if len(self.points) > 1:
                cv2.polylines(display_img, [np.array(self.points)], False, (0, 255, 0), 2)

            cv2.imshow(self.window_name, display_img)

            key = cv2.waitKey(30) & 0xFF  # ~30fps refresh
            if key == ord('c') and len(self.points) > 2:
                break
            elif key == ord('z'):
                self.points = self.points[:-30] if len(self.points) > 30 else []
            elif key == ord('r'):
                self.points = []
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

        cv2.destroyAllWindows()

        # --- Full Video Statistical Analysis ---
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array([self.points], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)

        all_h, all_s, all_v = [], [], []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        with Progress() as progress:
            task = progress.add_task("[cyan]Analyzing Video...", total=total_frames)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                flame_pixels = hsv[mask == 255]
                if flame_pixels.size > 0:
                    # Sampling to optimize memory
                    step = max(1, flame_pixels.shape[0] // 500)
                    all_h.extend(flame_pixels[::step, 0])
                    all_s.extend(flame_pixels[::step, 1])
                    all_v.extend(flame_pixels[::step, 2])
                progress.update(task, advance=1)

        cap.release()
        self.report_stats(all_h, all_s, all_v)

    def report_stats(self, h, s, v):
        h_data, s_data, v_data = np.array(h), np.array(s), np.array(v)
        console.print("\n" + "=" * 45)
        console.print("[bold green]VIDEO HSV STATISTICS[/bold green]")
        console.print("=" * 45)
        for name, data in [("Hue", h_data), ("Saturation", s_data), ("Value", v_data)]:
            console.print(f"\n[bold cyan]{name}:[/bold cyan]")
            console.print(f"  Range: {np.min(data)} - {np.max(data)} | Mean: {np.mean(data):.1f}")
            console.print(f"  P2 (Lower): {np.percentile(data, 2):.0f} | P98 (Upper): {np.percentile(data, 98):.0f}")

        console.print("\n[bold yellow]Recommended for main script:[/bold yellow]")
        console.print(
            f"lower = [{np.percentile(h_data, 2):.0f}, {np.percentile(s_data, 2):.0f}, {np.percentile(v_data, 5):.0f}]")
        console.print(f"upper = [{np.percentile(h_data, 98):.0f}, 255, 255]")


if __name__ == "__main__":
    # Check for CLI arg
    path = sys.argv[1] if len(sys.argv) > 1 else None

    # File Dialog if no arg
    if not path:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mts"), ("All", "*.*")])

    if path:
        FullVideoLassoStats(path).run_analysis()
    else:
        sys.exit()