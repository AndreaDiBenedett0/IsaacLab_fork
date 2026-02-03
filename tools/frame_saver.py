
import sys
sys.path.append("/home/user-05/isaac_lab_projects/IsaacLab_fork/scripts/tools")

from mp4_to_hdf5 import get_frames_from_mp4

import numpy as np
import imageio
import os

video_path = "/home/user-05/isaac_lab_projects/IsaacLab_fork/logs/rsl_rl/g1_flat_paper/2026-01-27_08-17-56/videos/play/fv_fd_vel06.mp4"
output_dir = "latex_frames"
os.makedirs(output_dir, exist_ok=True)

frames = get_frames_from_mp4(video_path)

N = 10
idx = np.linspace(1, len(frames)-1, N).astype(int)
for i, j in enumerate(idx):
    imageio.imwrite(os.path.join(output_dir, f"walk_{i:02d}.png"), frames[j])

print("Frame pronti per la tesi salvati in:", output_dir)