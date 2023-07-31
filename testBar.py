import time
from rich.progress import Progress, BarColumn, SpinnerColumn, TimeRemainingColumn, TimeElapsedColumn, DownloadColumn, \
    TransferSpeedColumn, TextColumn, track

description = "[#66CCFF]Loading"
task = track(range(100), description=description, style="white", complete_style="#66CCFF")
for p in task:
    time.sleep(0.02)


