# Multi-modal IMU and Wearable Camera Dataset — Processing Code

This repository provides code used to support the preparation of a dataset that combines:
- wearable inertial sensors (IMU), and
- wearable (shank-mounted) egocentric video.

The dataset itself is hosted separately (see the dataset link provided in the accompanying manuscript / data record).  
This GitHub repository is intended to document and share the **processing scripts** used for dataset preparation.

> **Repository structure (current):**
>
> - `Codes/` — scripts for dataset preparation and video/feature processing  
> - `README.md` — documentation (this file)

---

## What is inside `Codes/`

The `Codes/` directory contains scripts used for dataset preparation tasks such as:

- **Video anonymization** (e.g., face detection and blurring) to protect privacy before release  
- **Video-derived feature extraction** (e.g., optical-flow-based motion features) for downstream analysis

Please refer to the comments at the top of each script in `Codes/` for:
- required inputs,
- expected outputs,
- and how to run the script.

---

## Usage notes

- The scripts in this repository are provided as-is to support transparency and reproducibility.
- Users are **not** expected to uncomment hidden “alternative instructions.”  
  Any outdated or experimental code paths should be removed or converted into explicit options (e.g., flags or configuration values).

---

## Dependencies

The exact dependencies vary by script, but commonly used packages include:

- Python 3
- OpenCV (`cv2`)
- NumPy
- Pandas
- FFmpeg (command-line tool) — if re-encoding or audio removal is used
- Additional libraries may be required for specific pipelines (e.g., face detection)

Each script should specify its own requirements at the top of the file.

---

## How to run

Because scripts may have different inputs (local files vs cloud storage, etc.), please follow the usage instructions inside each script in `Codes/`.

A typical workflow is:

1. Place input files where the script expects them (or edit the input paths inside the script).
2. Run the script with Python:
   ```bash
   python <script_name>.py
