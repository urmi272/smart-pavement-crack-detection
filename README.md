# Smart Pavement Crack Detection using Computer Vision

A computer vision project that detects cracks in road surface images using **classical image processing techniques** - no deep learning required. Built with Python and OpenCV as a university course project.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Problem Statement

Road surface damage such as cracks can lead to accidents and infrastructure deterioration. 
This project proposes an automated approach for detecting road cracks using classical image processing techniques implemented in Python and OpenCV.

- **Accidents** - especially for two-wheelers and cyclists
- **Vehicle damage** - tyres, suspension, and alignment issues
- **Higher maintenance costs** for municipal bodies due to delayed repairs
- **Flooding** - cracks allow water seepage, weakening the road foundation

Manual inspection of roads is **slow, expensive, and inconsistent**. An automated image-based crack detection system can help municipal authorities quickly identify damaged sections and prioritise repairs.

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Image Loading** | Load single image or batch-process an entire folder |
| 2 | **Grayscale Conversion** | Convert colour images to grayscale for processing |
| 3 | **Noise Removal** | Gaussian Blur to smooth out sensor noise |
| 4 | **Contrast Enhancement** | CLAHE for better crack visibility |
| 5 | **Edge Detection** | Canny Edge Detection to find crack boundaries |
| 6 | **Thresholding** | Adaptive thresholding for robust binarisation |
| 7 | **Morphological Ops** | Dilation, Erosion, Opening & Closing to clean results |
| 8 | **Contour Detection** | Identify and filter crack regions |
| 9 | **Crack Highlighting** | Draw detected cracks in red on the original image |
| 10 | **Severity Classification** | LOW / MEDIUM / HIGH based on crack area coverage |
| 11 | **Visual Pipeline** | Side-by-side display of all intermediate stages |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV (`cv2`) |
| Numerical Computing | NumPy |
| Visualisation | Matplotlib |

---

## Folder Structure
---
crackvision-road-analysis
│
├── data
│   ├── road_test1.jpg
│   └── road_test2.jpg
│
├── outputs
│   ├── road1_gray.jpg
│   ├── road1_blur.jpg
│   ├── road1_edges.jpg
│   ├── road1_morphology.jpg
│   └── road1_result.jpg
│
├── src
│   ├── main.py
│   ├── preprocessing.py
│   ├── detection.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── project_overview.txt
---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/urmi272/smart-pavement-crack-detection.git
   cd crackvision-road-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```

   Activate the environment:

   On Windows
   ```bash
   .venv\Scripts\activate
   ```

   On macOS / Linux
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Process all images in `data/` folder
```bash
python src/main.py
```

### Process a single image
```bash
python src/main.py data/sample1.jpg
```

### What happens when you run it

1. The system loads each image from `data/`
2. Applies the complete crack detection pipeline
3. Displays intermediate stages in a Matplotlib window
4. Saves all outputs (grayscale, edges, result, etc.) to the `outputs/` folder
5. Prints a **severity report** to the console

---

### Pipeline Stages

The system generates a side-by-side visualisation showing:

| Stage | Description |
|-------|-------------|
| Original | Input road image |
| Grayscale | Converted to single channel |
| Blurred | After Gaussian noise removal |
| Canny Edges | Edge-detected binary image |
| Threshold | Adaptive thresholding result |
| Morphology | After morphological cleanup |
| Detected Cracks | Final result with cracks in red |

---

## How It Works

```
Input Image
    │
    ▼
Grayscale Conversion
    │
    ▼
CLAHE Contrast Enhancement
    │
    ▼
Gaussian Blur (noise removal)
    │
    ├──────────────────┐
    ▼                  ▼
Canny Edge         Adaptive
Detection          Thresholding
    │                  │
    └────── OR ────────┘
           │
           ▼
   Morphological Operations
   (Close → Open → Dilate)
           │
           ▼
   Contour Detection & Filtering
           │
           ▼
   Draw Cracks + Severity Report
```

---

## Severity Classification

| Level | Crack Area | Meaning |
|-------|-----------|---------|
| LOW | < 1% | Minor surface cracks, road is mostly safe |
| MEDIUM | 1% - 5% | Moderate damage, maintenance recommended |
| HIGH | > 5% | Severe cracking, immediate repair needed |

---

## License

This project is for educational purposes as part of a university course submission.

---

## Author

*Developed by: Urmi Barman*
*Course: Computer Vision Course*  
*VIT BHOPAL UNIVERSITY*  
