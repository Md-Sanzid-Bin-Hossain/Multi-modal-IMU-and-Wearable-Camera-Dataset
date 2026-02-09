# Wearable Motion Capture Dataset for Gait Analysis  
### IMUs and Shank-Mounted Egocentric Cameras — Processing Code

This repository provides the **processing and preparation code** used in the paper:

> **A Wearable Motion Capture Dataset for Gait Analysis Using IMUs and Shank-Mounted Egocentric Cameras**  
> Md Sanzid Bin Hossain *et al.*

The **dataset itself is hosted separately** and can be accessed via the link provided in the manuscript and data record.  
This GitHub repository is intended to support **transparency, reproducibility, and reuse** by sharing the scripts used during dataset preparation and feature extraction.  [oai_citation:0‡Main_manuscript.pdf](sediment://file_000000008fc0722fb07eff7325175072)

---

## Overview of the Dataset

This work introduces a **multimodal wearable motion capture dataset** designed to support both **biomechanics** and **machine learning** research on human gait.

The dataset includes synchronized data from:
- **Wearable inertial measurement units (IMUs)** mounted on the lower limbs
- **Shank-mounted egocentric cameras** capturing first-person lower-limb video
- **Ground-truth joint kinematics** derived from optical motion capture and musculoskeletal modeling

Participants performed a wide range of locomotion tasks, including:
- Overground walking (multiple speeds, turning, obstacles)
- Treadmill walking (four speed conditions)
- Slope ascent and descent
- Stair ascent and descent

## Locomotion Trial Types

<p align="center">
  <img src="images/Trial_types.png" width="500">
</p>
<p align="center">
  <em>Different trial types in the dataset: (a) stair (2 repetitions), (b) slope (2 repetitions), (c) treadmill (4 speeds), (d) overground (4 speeds), (e) overground round, and (f) overground obstacles.</em>
</p>


<p align="center">
  <img src="images/marker_and_sensors.png" width="700">
</p>
<p align="center">
  <em>Markers, IMUs, and camera location. Primary markers were placed on anatomical bony landmarks following a modified Helen Hayes marker set to ensure consistency across participants. Additional markers were applied on soft tissue regions to assist with the kinematics tracking, with placement adjusted as needed based on reference anatomical photographs captured during the static calibration pose. T_head, R_hip, L_hip markers are used for scaling purposes and are not used directly for kinematics tracking.</em>
</p>

The combination of IMU signals, egocentric video, and joint kinematics enables research on:
- Joint kinematics estimation and prediction
- Multimodal sensor fusion
- Locomotion mode recognition
- Gait event detection
- Anticipatory control and intent recognition

---

## Example Dataset Structure

The released dataset is organized **per participant**, with subfolders corresponding to locomotion modes and sensor modalities.

**Illustrative structure (simplified):**
