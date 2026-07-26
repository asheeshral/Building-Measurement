# 🏢 Satellite Building Length & Width Measurement System

An end-to-end Python system for automated detection, measurement, and visualization of building dimensions (Length, Width, and Footprint Area) in high-resolution satellite imagery (SpaceNet 2 dataset).

---

## 🌟 Key Features

- **Automated Geometry Extraction**: Parses building polygon coordinates from WKT (Well-Known Text) annotations.
- **Minimum Bounding Box Measurement**: Computes oriented minimum area rectangles (`cv2.minAreaRect`) to determine physical length and width independent of building orientation.
- **Physical Scale Conversion**: Translates pixel dimensions into ground meters using Ground Sample Distance (GSD = 0.30 m/pixel).
- **Exact Area Calculation**: Calculates true building polygon footprint area ($\text{m}^2$) using Shapely geometry engine.
- **Rich Visual Output**:
  - Raw original satellite imagery export.
  - Intermediate building footprint overlays.
  - Final annotated output displaying building index, dimensions ($L \times W$), and pixel metrics.
  - Side-by-side multi-panel comparison plots.

---

## 📸 Output & Visual Results

### 1. Side-by-Side Comparison Plot
![Building Measurement Plot](images/building_measurement_plot.png)

### 2. Output Image Breakdown

| Original Satellite Image | Building Footprints | Measured Output |
|:-----------------------:|:-------------------:|:---------------:|
| ![Original Satellite Image](images/original_satellite_image.png) | ![Input Image](images/measured_building_input.png) | ![Measured Output](images/measured_building_output.png) |

---

## 📊 Measurement Sample Output

```text
==================================================
SpaceNet 2 Building Length & Width Measurement
==================================================
[Loaded Image]: RGB-PanSharpen_AOI_3_Paris_img485.tif (ID: AOI_3_Paris_img485)
[Building Count]: 45 building footprints found.

--- Building Measurement Results ---
Building #1:
  Length : 44.13 pixels -> 13.24 meters
  Width  : 14.62 pixels -> 4.39 meters
  Actual Footprint Area : 327.56 px² -> 29.48 m²

Building #2:
  Length : 16.95 pixels -> 5.08 meters
  Width  : 10.24 pixels -> 3.07 meters
  Actual Footprint Area : 86.78 px² -> 7.81 m²

Building #3:
  Length : 40.40 pixels -> 12.12 meters
  Width  : 22.81 pixels -> 6.84 meters
  Actual Footprint Area : 759.60 px² -> 68.36 m²
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

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Place your SpaceNet 2 dataset directory or satellite `.tif` images & `.csv` WKT solution files in the dataset path.
2. Run the measurement script:
```bash
python measure_building.py
```
3. All output images and comparison plots will be saved inside the `images/` directory automatically.

---

## 🧮 How It Works

1. **Polygon Parsing**: Loads WKT polygon vertices from `summaryData/AOI_3_Paris_Train_Building_Solutions.csv`.
2. **Oriented Rectangles**: Constructs standard minimum bounding rectangle using OpenCV to capture arbitrary building orientations:
   $$\text{Length}_{\text{m}} = \max(\text{side}_1, \text{side}_2) \times \text{GSD}$$
   $$\text{Width}_{\text{m}} = \min(\text{side}_1, \text{side}_2) \times \text{GSD}$$
3. **Footprint Area**:
   $$\text{Area}_{\text{m}^2} = \text{Polygon Area}_{\text{px}} \times \text{GSD}^2$$
4. **Drawing Annotations**: Overlays green polygon outlines, cyan rotated bounding boxes, and high-visibility text cards over each detected building.

---

## 📄 License
MIT License
