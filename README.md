# 🏢 Satellite 3D Building Measurement System (Length, Width & Height)

An end-to-end Python system for automated 3D detection, measurement, and visualization of building dimensions (**Length**, **Width**, **Height**, **Story/Floor Count**, and **3D Footprint Volume**) in high-resolution satellite imagery (SpaceNet 2 dataset).

---

## 🌟 Key Features

- **Interactive Native Windows File Selector**: Whenever you run `python measure_building.py`, a Windows Explorer pop-up window opens allowing you to select **any `.tif`, `.png`, or `.jpg`** satellite or mask image dynamically!
- **Dynamic Dataset Footprint Lookup**: Automatically parses image filenames (e.g. `AOI_3_Paris_img123`) and retrieves matching WKT building footprints from CSV datasets.
- **Automated Mask Contour Extraction**: If no CSV footprint exists for the selected image, automatically extracts polygon footprints directly from image contours (`cv2.findContours`).
- **Minimum Bounding Box Measurement**: Computes oriented minimum area rectangles (`cv2.minAreaRect`) to determine physical length and width independent of building orientation.
- **Physics-Based & Morphological Height Estimation**: Combines local shadow displacement gradient analysis along the solar illumination vector with morphological footprint scaling models to calculate building height ($H_{\text{m}}$).
- **Floor Count & 3D Volumetric Metrics**: Estimates total story count ($F$) and computes complete 3D structure volume ($V_{\text{m}^3}$).
- **Physical Scale Conversion**: Translates pixel dimensions into ground meters using Ground Sample Distance ($\text{GSD} = 0.30\text{ m/pixel}$).
- **High Performance & Sub-Crop Optimization**: Fast localized sub-crop processing delivering calculation speeds of **~1.75 ms per building**.
- **3D Wireframe Spatial Visualization**:
  - Overlays 2D ground footprint polygon outlines (green).
  - Draws minimum area oriented bounding boxes (cyan).
  - Renders 3D isometric wireframe prism projections (magenta/pink extrusions) showing true spatial elevation.
  - Generates a 4-panel analytical comparison dashboard including distribution bar charts of building height vs 3D volume.

---

## 📸 Output & Visual Results

### 1. 4-Panel Analytics & 3D Extrusion Dashboard Plot
![Building Measurement Plot](images/building_measurement_plot.png)

### 2. Output Image Breakdown

| Original Satellite Image | Building Ground Footprints | 3D Measured Output (L x W x H + Wireframes) |
|:-----------------------:|:-------------------------:|:------------------------------------------:|
| ![Original Satellite Image](images/original_satellite_image.png) | ![Input Image](images/measured_building_input.png) | ![Measured Output](images/measured_building_output.png) |

---

## 🚀 How to Run & Select Images

### Option A: Interactive Windows Explorer File Picker (Recommended)
Simply type in your terminal:
```bash
python measure_building.py
```
A Windows Explorer file dialog pop-up will appear automatically. Browse and select **any `.tif` or `.png` satellite image** from your computer!

### Option B: Pass Specific Image File Path via Command Line
```bash
python measure_building.py --image "dataset/AOI_3_Paris_Train/AOI_3_Paris_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_3_Paris_img200.tif"
```

---

## 📊 Measurement Sample Output

```text
==================================================
SpaceNet 2 Dynamic 3D Building Measurement
 (Length, Width, Height, Floors & Volume)
==================================================
Opening Windows Explorer file dialog to select a satellite image...
[User Selected Image]: C:\...\RGB-PanSharpen_AOI_3_Paris_img200.tif
[Loaded Image & Polygons in 15.2ms]: RGB-PanSharpen_AOI_3_Paris_img200.tif (ID: AOI_3_Paris_img200)
[Building Count]: 38 building footprints found.

--- Building 3D Measurement Results ---
Building #1:
  Length   : 44.13 px  -> 13.24 m
  Width    : 14.62 px  -> 4.39 m
  Height   : 28.32 px  -> 8.50 m (3 Floors)
  Footprint: 327.56 px² -> 29.48 m²
  3D Volume: 250.51 m³
  ----------------------------------------
```

---

## 🛠️ Installation & Setup Steps

### Prerequisites
- Python 3.8 or higher
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/asheeshral/building-measurement.git
cd building-measurement
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 📄 License
MIT License
