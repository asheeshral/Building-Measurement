import cv2
import csv
import os
import sys
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Polygon

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def load_sample_building(csv_path, img_dir, target_img_id=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    image_polygons = {}
    with open(csv_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            img_id = row['ImageId']
            pix_wkt = row['PolygonWKT_Pix']
            
            if pix_wkt != 'POLYGON EMPTY':
                geom = wkt.loads(pix_wkt)
                coords = [(pt[0], pt[1]) for pt in geom.exterior.coords]
                if img_id not in image_polygons:
                    image_polygons[img_id] = []
                image_polygons[img_id].append(coords)

    if target_img_id is None or target_img_id not in image_polygons:
        for img_id, polys in image_polygons.items():
            if len(polys) >= 5:
                target_img_id = img_id
                break

    img_filename = f"RGB-PanSharpen_{target_img_id}.tif"
    img_path = os.path.join(img_dir, img_filename)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Satellite image file not found: {img_path}")

    image = tiff.imread(img_path)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, image_polygons[target_img_id], img_path, target_img_id


def calculate_building_dimensions(polygon_coords, gsd_meters_per_pixel=0.30):
    pts = np.array(polygon_coords, dtype=np.float32).reshape((-1, 1, 2))

    # Minimum Area Rectangle
    rect = cv2.minAreaRect(pts)
    center, (side1, side2), angle = rect

    length_px = max(side1, side2)
    width_px = min(side1, side2)

    length_m = length_px * gsd_meters_per_pixel
    width_m = width_px * gsd_meters_per_pixel

    # Actual building footprint area using Shapely
    polygon = Polygon(polygon_coords)
    area_px = polygon.area
    area_m = area_px * (gsd_meters_per_pixel ** 2)

    box_pts = cv2.boxPoints(rect)
    box_pts = np.int32(box_pts)

    return {
        'rect': rect,
        'box_points': box_pts,
        'center': (int(center[0]), int(center[1])),
        'length_px': length_px,
        'width_px': width_px,
        'length_m': length_m,
        'width_m': width_m,
        'area_px': area_px,
        'area_m': area_m,
        'angle': angle
    }


def draw_building_measurements(image, polygon_coords, measurement_info, building_index=1):
    annotated = image.copy()
    
    poly_pts = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    box_pts = measurement_info['box_points']
    cv2.polylines(annotated, [box_pts], isClosed=True, color=(255, 255, 0), thickness=2)

    l_m = measurement_info['length_m']
    w_m = measurement_info['width_m']
    l_px = measurement_info['length_px']
    w_px = measurement_info['width_px']
    
    label_str = f"B{building_index}: {l_m:.1f}m x {w_m:.1f}m ({l_px:.0f}px x {w_px:.0f}px)"

    cx, cy = measurement_info['center']

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)
    
    text_x = max(10, cx - text_w // 2)
    text_y = max(20, cy - 5)

    cv2.rectangle(
        annotated, 
        (text_x - 3, text_y - text_h - 4), 
        (text_x + text_w + 3, text_y + baseline + 2), 
        (0, 0, 0), 
        -1
    )
    
    cv2.putText(
        annotated, 
        label_str, 
        (text_x, text_y), 
        font, 
        font_scale, 
        (255, 255, 255), 
        thickness, 
        lineType=cv2.LINE_AA
    )

    return annotated


def save_image(path, img):
    ext = os.path.splitext(path)[1]
    success, encoded_img = cv2.imencode(ext, img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded_img.tobytes())
        return True
    return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    base_dir = os.path.join(script_dir, r"dataset\AOI_3_Paris_Train\AOI_3_Paris_Train")
    csv_path = os.path.join(base_dir, r"summaryData\AOI_3_Paris_Train_Building_Solutions.csv")
    img_dir = os.path.join(base_dir, r"RGB-PanSharpen")
    
    original_sat_path = os.path.join(images_dir, "original_satellite_image.png")
    input_path = os.path.join(images_dir, "measured_building_input.png")
    output_path = os.path.join(images_dir, "measured_building_output.png")
    plot_path = os.path.join(images_dir, "building_measurement_plot.png")

    print("==================================================")
    print("SpaceNet 2 Building Length & Width Measurement")
    print("==================================================")

    image, polygons, img_path, img_id = load_sample_building(csv_path, img_dir)
    print(f"[Loaded Image]: {os.path.basename(img_path)} (ID: {img_id})")
    print(f"[Building Count]: {len(polygons)} building footprints found.")

    # Save original satellite image explicitly
    save_image(original_sat_path, image)
    print(f"[Saved Original Satellite Image]: {original_sat_path}")

    # Save input copy
    save_image(input_path, image)
    print(f"[Saved Input Image]: {input_path}")

    # Draw footprints ground-truth overlay
    footprints_img = image.copy()
    for poly in polygons:
        poly_pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(footprints_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    annotated_img = image.copy()
    
    print("\n--- Building Measurement Results ---")
    for idx, poly in enumerate(polygons, 1):
        dim = calculate_building_dimensions(poly, gsd_meters_per_pixel=0.30)
        
        # Skip tiny annotation artifacts
        if dim['area_m'] < 1.0:
            continue

        print(f"Building #{idx}:")
        print(f"  Length : {dim['length_px']:.2f} pixels -> {dim['length_m']:.2f} meters")
        print(f"  Width  : {dim['width_px']:.2f} pixels -> {dim['width_m']:.2f} meters")
        print(f"  Actual Footprint Area : {dim['area_px']:.2f} px² -> {dim['area_m']:.2f} m²")

        annotated_img = draw_building_measurements(
            annotated_img,
            poly,
            dim,
            building_index=idx
        )

    save_image(output_path, annotated_img)
    print(f"\n[Saved Annotated Output Image]: {output_path}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"1. Original Satellite Image\n({img_id})", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(footprints_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"2. Building Footprints\n(Green outlines)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"3. Building Length & Width Measurement\n(1 px = 0.30m)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"[Saved Matplotlib Plot]: {plot_path}")
    print("\nExecution completed successfully!")

if __name__ == '__main__':
    main()
