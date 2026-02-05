"""
Dense Optical Flow Feature Extraction (Histogram of Optical Flow)

This script computes dense optical flow between consecutive frames of a video
using Farnebäck’s algorithm and summarizes motion using a histogram of flow
orientations weighted by magnitude (HOF-style descriptor).

For each input video:
  1. Dense optical flow is computed between consecutive grayscale frames.
  2. Each flow field is converted into a fixed-length histogram of orientations.
  3. The resulting per-frame histograms are saved as a CSV file.

Dependencies:
  - OpenCV
  - NumPy
  - Pandas
"""

import cv2
import numpy as np
import pandas as pd
import os


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VIDEO_FILES = [
    "video/subject 10/P002_T010_overground_round_left.mp4"
]

OUTPUT_DIR = "hof_features/left"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ORIENTATION_BINS = 18   # Number of orientation bins for HOF
FRAME_HEIGHT = 1080      # Used for normalization
FRAME_WIDTH = 1920


# ---------------------------------------------------------------------
# Optical Flow Histogram Descriptor
# ---------------------------------------------------------------------

class OpticalFlowHistogram:
    """
    Computes a histogram of optical flow orientations weighted by magnitude.
    """

    def __init__(self, flow, num_bins):
        """
        Parameters
        ----------
        flow : np.ndarray
            Optical flow field of shape (H, W, 2)
        num_bins : int
            Number of orientation bins
        """
        self.flow = flow
        self.num_bins = num_bins
        self.angle_unit = 360.0 / num_bins

    def extract(self):
        """
        Returns
        -------
        hist : np.ndarray
            Normalized histogram of optical flow orientations
        """
        mag, ang = cv2.cartToPolar(
            self.flow[..., 0],
            self.flow[..., 1],
            angleInDegrees=True
        )

        bin_idx = (ang // self.angle_unit).astype(int)
        hist = np.zeros(self.num_bins, dtype=np.float32)

        for b in range(self.num_bins):
            hist[b] = mag[bin_idx == b].sum()

        # Normalize by image area for scale invariance
        hist /= (FRAME_HEIGHT * FRAME_WIDTH)
        return hist


# ---------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------

for video_path in VIDEO_FILES:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()

    if not ret:
        raise RuntimeError("Failed to read the first frame.")

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    flow_fields = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flow_fields.append(flow)
        frame_idx += 1
        print(f"{video_path} | Frame {frame_idx}/{total_frames - 1}")

        prev_gray = curr_gray

    cap.release()

    # -----------------------------------------------------------------
    # Feature Extraction
    # -----------------------------------------------------------------

    histograms = []
    for flow in flow_fields:
        hof = OpticalFlowHistogram(flow, N_ORIENTATION_BINS)
        histograms.append(hof.extract())

    df = pd.DataFrame(histograms)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{video_name}.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved features to: {output_path}")
