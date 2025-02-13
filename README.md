# Anomalib OpenVINO C++ Reference

This repository provides a C++ implementation of anomaly detection inference using OpenVINO, based on Anomalib. Currently, only the **Padim** algorithm has been validated, and other model algorithms have not yet been tested.

## Versions

### Fast Inference (inference.cpp)
- **Approach:** Performs anomaly detection by predicting per-pixel anomaly scores.
- **Performance:** Verified that on 256x256 images, the GPU (integrated graphics) is at least 10ms faster than the CPU.
- **Note:** In fast mode, the `imread` function accounts for nearly half of the total processing time. For production environments, consider preloading images into memory to reduce I/O overhead.
- **Summary:** Delivers fast inference with good accuracy at a 256x256 resolution.

### Visualized Inference (inference_with_visuals.cpp)
- **Functionality:** Enhances the basic inference by overlaying segmentation results and heatmaps on the original image to visualize anomalies.
- **Purpose:** Facilitates enhanced analysis and verification of detection outcomes.

## Features
- **Efficient Inference:** Utilizes OpenVINO for high-speed anomaly detection.
- **Per-Pixel Detection:** Supports fine-grained, per-pixel anomaly detection.
- **Enhanced Visualization:** Provides heatmaps and segmentation overlays for improved analysis.

## Notice
- **Demo Example:** This project serves as a demonstration. Users are expected to integrate their own models and set appropriate thresholds, which are available in the exported metadata JSON.
- **Heatmap Rendering:** The current heatmap rendering algorithm may differ from the official version. While the visual output is similar, the rendering speed is slower. Contributions and suggestions for optimizing heatmap generation are welcome.

## Contributions
Contributions, suggestions, and improvements are welcome! Please feel free to open an issue or submit a pull request.

