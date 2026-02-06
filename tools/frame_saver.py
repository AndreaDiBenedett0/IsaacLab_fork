
import sys
sys.path.append("/home/user-05/isaac_lab_projects/IsaacLab_fork/scripts/tools")

from mp4_to_hdf5 import get_frames_from_mp4

import numpy as np
import imageio
import os

video_path = "/home/user-05/isaac_lab_projects/IsaacLab_fork/logs/rsl_rl/g1_flat_paper/2026-02-03_08-32-06/videos/play/rl-video-step-0.mp4"
output_dir = "latex_frames"
os.makedirs(output_dir, exist_ok=True)

frames = get_frames_from_mp4(video_path)
total_frames = len(frames)

# -------------------------------------------
# Calcola i limiti dei quarti
# -------------------------------------------
quarter = total_frames // 4

# Indici di inizio e fine per ogni quarto
quarters = [
    (0, quarter),
    (quarter, 2 * quarter),
    (2 * quarter, 3 * quarter),
    (3 * quarter, total_frames)
]

N = 20  # frame da estrarre per ogni quarto

# -------------------------------------------
# Estrai N frame uniformemente per ogni quarto
# -------------------------------------------
for q_idx, (start, end) in enumerate(quarters):
    segment = frames[start:end]
    
    idx = np.linspace(1, len(segment) - 1, N).astype(int)
    
    for i, j in enumerate(idx):
        imageio.imwrite(
            os.path.join(output_dir, f"walk_q{q_idx}_frame_{i:02d}.png"),
            segment[j]
        )

print("Frame estratti per ogni quarto salvati in:", output_dir)