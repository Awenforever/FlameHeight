# -*- coding: utf-8 -*-
# @Time    : 2025/12/10 22:00
# @Author  : Jinhong Wu
# @File    : utils.py
# @Project : Develop
# Copyright (c) 2025 Jinhong Wu. All rights reserved.

# rich
from rich.console import Console
from rich.progress import Progress
from rich.progress import (
    SpinnerColumn,
    TextColumn,
    BarColumn,
    ProgressColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.text import Text

class SpeedColumn(ProgressColumn):
    def render(self, task):
        if not task.elapsed or task.elapsed <= 0:
            return Text("-- fps", style="cyan")
        speed = task.completed / task.elapsed
        return Text(f"{speed:2.2f} fps", style="cyan")


console = Console()
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TaskProgressColumn("[progress.percentage]{task.percentage:>3.2f}%"),
    SpeedColumn(),
    TimeRemainingColumn(),
    console=console,
    expand=True,
    speed_estimate_period=60 * 60  # seconds
)
