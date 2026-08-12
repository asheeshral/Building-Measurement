import cv2
import csv
import os
import sys
import math
import time
import argparse
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


def select_image_file_dialog(initial_dir=None):
    """
    Opens a native Windows Explorer file dialog allowing the user to select any satellite image.
    Returns selected file path or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        if initial_dir is None or not os.path.exists(initial_dir):
            initial_dir = os.getcwd()

        file_path = filedialog.askopenfilename(
            title="Select Satellite Image for 3D Building Measurement",
            initialdir=initial_dir,
            filetypes=[
                ("Supported Images (*.tif, *.png, *.jpg)", "*.tif;*.tiff;*.png;*.jpg;*.jpeg"),
                ("GeoTIFF Images (*.tif, *.tiff)", "*.tif;*.tiff"),
                ("PNG Images (*.png)", "*.png"),
                ("JPEG Images (*.jpg, *.jpeg)", "*.jpg;*.jpeg"),
                ("All Files (*.*)", "*.*")
            ]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[GUI File Dialog Notice]: {e}")
        return None


def load_image_file(img_path):
    """
    Universally loads satellite or aerial images supporting .png, .jpg, .jpeg, .tif, .tiff formats.
    Robustly converts 16-bit GeoTIFF imagery to 8-bit RGB using percentile normalization.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    ext = os.path.splitext(img_path)[1].lower()
    if ext in ['.tif', '.tiff']:
        image = tiff.imread(img_path)
    else:
        image = cv2.imread(img_path)

    if image is None:
        raise ValueError(f"Failed to read image at: {img_path}")

    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        if image.max() > 255:
            p_lo, p_hi = np.percentile(image, (1, 99))
            if p_hi > p_lo:
                image = np.clip((image.astype(np.float32) - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
            else:
                image = (image.astype(np.float32) / image.max() * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if len(image.shape) == 3:
        if ext in ['.tif', '.tiff']:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def extract_image_id_from_path(img_path):
    """
    Extracts SpaceNet ImageId (e.g. 'AOI_3_Paris_img160') from filename or path.
    Handles filenames like:
      - 'RGB-PanSharpen_AOI_3_Paris_img160.tif' -> 'AOI_3_Paris_img160'
      - 'MS_AOI_3_Paris_img160.tif' -> 'AOI_3_Paris_img160'
      - 'AOI_3_Paris_img160.png' -> 'AOI_3_Paris_img160'
    """
    filename = os.path.basename(img_path)
    name = os.path.splitext(filename)[0]

    prefixes = [
        "RGB-PanSharpen_",
        "PAN_",
        "MS_",
        "3Band_",
        "RGB_"
    ]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
            break

    return name


def find_csv_file(csv_path, script_dir):
    """
    Locates the SpaceNet ground-truth CSV file reliably across directory structures.
    """
    candidates = [
        csv_path,
        os.path.join(script_dir, "dataset", "AOI_3_Paris_Train", "AOI_3_Paris_Train", "summaryData", "AOI_3_Paris_Train_Building_Solutions.csv"),
        os.path.join(script_dir, "dataset", "summaryData", "AOI_3_Paris_Train_Building_Solutions.csv"),
        os.path.join(script_dir, "AOI_3_Paris_Train_Building_Solutions.csv")
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c

    for root, _, files in os.walk(script_dir):
        for f in files:
            if f.endswith(".csv") and "Building_Solutions" in f:
                return os.path.join(root, f)
    return None


def debug_and_load_polygons_from_csv(csv_path, target_img_id):
    """
    Step 1 & Step 2: Debugs and loads all ground-truth polygons matching target_img_id from SpaceNet CSV.
    Ensures exact ImageId matching without false partial-substring overlaps.
    """
    if not csv_path or not os.path.exists(csv_path):
        return [], 0, 0, 0

    polygons = []
    normalized_target = extract_image_id_from_path(target_img_id)

    matching_rows = 0
    non_empty_count = 0
    parsed_count = 0

    try:
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                img_id = row.get('ImageId', '').strip()
                norm_row_id = extract_image_id_from_path(img_id)

                # Strict exact match (prevent 'AOI_3_Paris_img160' matching 'img1601', 'img1603', etc.)
                if norm_row_id == normalized_target or img_id == target_img_id:
                    matching_rows += 1
                    pix_wkt = row.get('PolygonWKT_Pix', '').strip()
                    if pix_wkt and pix_wkt != 'POLYGON EMPTY':
                        non_empty_count += 1
                        try:
                            geom = wkt.loads(pix_wkt)
                            if hasattr(geom, 'exterior') and geom.exterior:
                                coords = [(float(pt[0]), float(pt[1])) for pt in geom.exterior.coords]
                                if len(coords) >= 3:
                                    polygons.append(coords)
                                    parsed_count += 1
                        except Exception as parse_err:
                            print(f"[CSV Debug Warning]: Failed to parse WKT row #{matching_rows}: {parse_err}")
    except Exception as e:
        print(f"[CSV Debug Error]: Failed to read CSV file: {e}")

    return polygons, matching_rows, non_empty_count, parsed_count


def clip_polygon_to_image(polygon_coords, img_w, img_h):
    """
    Step 6: Clips polygon coordinates to image boundary [0, img_w] x [0, img_h] for safe visualization.
    """
    clipped = []
    for x, y in polygon_coords:
        cx = max(0.0, min(float(img_w), x))
        cy = max(0.0, min(float(img_h), y))
        clipped.append((cx, cy))
    return clipped


def generate_diagnostic_gt_image(image, polygons, output_path, img_id):
    """
    Step 4: Generates debug_all_spacenet_polygons.png overlaying EVERY ground-truth CSV polygon
    with explicit GT-1, GT-2, ..., GT-N labels.
    """
    debug_img = image.copy()
    for idx, poly in enumerate(polygons, 1):
        poly_pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        label = f"GT-{idx}"
        cv2.rectangle(debug_img, (cx - 2, cy - 14), (cx + 45, cy + 4), (0, 0, 0), -1)
        cv2.putText(debug_img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    save_image(output_path, debug_img)
    print(f"[Saved Diagnostic GT Overlay]: {output_path}")


def detect_buildings_from_aerial_image(image, gsd=0.30):
    """
    MODE 2: Computer Vision Fallback Building Detector (for images without SpaceNet CSV annotations).
    """
    if image is None:
        return []

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    veg_mask = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([85, 255, 255]))

    b, g, r = cv2.split(image.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_mask = (exg > 20.0).astype(np.uint8) * 255
    veg_mask = cv2.bitwise_or(veg_mask, exg_mask)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shadow_mask = (gray < 35).astype(np.uint8) * 255

    blurred = cv2.bilateralFilter(image, 7, 50, 50)
    blurred_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred_gray)

    candidate_masks = []
    _, m1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate_masks.append(m1)

    _, m2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidate_masks.append(m2)

    q65, q80 = np.percentile(enhanced, [65, 80])
    _, m3 = cv2.threshold(enhanced, int(q65), 255, cv2.THRESH_BINARY)
    _, m4 = cv2.threshold(enhanced, int(q80), 255, cv2.THRESH_BINARY)
    candidate_masks.extend([m3, m4])

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    min_area_m2 = 15.0
    max_area_m2 = min(3500.0, (img_area * (gsd ** 2)) * 0.15)

    all_candidates = []

    for mask in candidate_masks:
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
        m[veg_mask > 0] = 0
        m[shadow_mask > 0] = 0

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area_px = cv2.contourArea(cnt)
            area_m2 = area_px * (gsd ** 2)

            if area_m2 < min_area_m2 or area_m2 > max_area_m2:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area_px / hull_area
            if solidity < 0.70:
                continue

            rect = cv2.minAreaRect(cnt)
            s1, s2 = rect[1]
            l_m = max(s1, s2) * gsd
            w_m = min(s1, s2) * gsd
            if w_m < 2.5:
                continue

            aspect_ratio = l_m / max(0.1, w_m)
            if aspect_ratio > 4.5:
                continue

            extent = area_px / max(1.0, s1 * s2)
            if extent < 0.45:
                continue

            eps = 0.018 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            coords = [(float(p[0][0]), float(p[0][1])) for p in approx]
            if len(coords) < 4 or len(coords) > 12:
                continue

            c_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(c_mask, [np.int32(coords)], 255)
            veg_in_poly = np.count_nonzero(veg_mask[c_mask == 255])
            total_poly_px = np.count_nonzero(c_mask == 255)
            if total_poly_px == 0:
                continue
            veg_ratio = veg_in_poly / float(total_poly_px)
            if veg_ratio > 0.20:
                continue

            score = solidity * extent * (1.0 - veg_ratio)
            try:
                poly_geom = Polygon(coords)
                if poly_geom.is_valid and poly_geom.area > 0:
                    all_candidates.append((poly_geom, coords, area_m2, l_m, w_m, score))
            except Exception:
                pass

    all_candidates.sort(key=lambda x: x[5], reverse=True)
    final_polygons = []

    for cand in all_candidates:
        p_geom, coords, area_m2, l_m, w_m, score = cand
        overlap = False
        for f_geom, _, _, _, _, _ in final_polygons:
            try:
                inter_area = p_geom.intersection(f_geom).area
                union_area = p_geom.union(f_geom).area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > 0.30 or inter_area / p_geom.area > 0.50:
                    overlap = True
                    break
            except Exception:
                pass
        if not overlap:
            final_polygons.append(cand)

    output_coords = [cand[1] for cand in final_polygons]
    print(f"[Fallback CV Detector]: Found {len(output_coords)} high-confidence building footprints.")
    return output_coords


def load_building_data(csv_path, default_img_dir, selected_img_path=None):
    """
    Step 7: Implements clear mode separation (MODE 1: Ground Truth vs MODE 2: Fallback CV).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    valid_csv_path = find_csv_file(csv_path, script_dir)

    image = None
    polygons = []
    img_id = "Custom_Image"
    is_spacenet_gt = False
    debug_info = {}

    if selected_img_path and os.path.exists(selected_img_path):
        img_path = selected_img_path
        img_id = extract_image_id_from_path(img_path)
        image = load_image_file(img_path)

        if valid_csv_path:
            polygons, matching_rows, non_empty_count, parsed_count = debug_and_load_polygons_from_csv(valid_csv_path, img_id)
            debug_info = {
                'matching_rows': matching_rows,
                'non_empty_count': non_empty_count,
                'parsed_count': parsed_count
            }
            if polygons:
                is_spacenet_gt = True

        if not polygons:
            ext = os.path.splitext(img_path)[1].lower()
            gsd = 0.30 if ext in ['.tif', '.tiff'] else 0.50
            polygons = detect_buildings_from_aerial_image(image, gsd=gsd)

        return image, polygons, img_path, img_id, is_spacenet_gt, debug_info

    # Default fallback image selection
    if valid_csv_path:
        image_polygons = {}
        with open(valid_csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                i_id = row.get('ImageId', '').strip()
                pix_wkt = row.get('PolygonWKT_Pix', '').strip()
                if pix_wkt and pix_wkt != 'POLYGON EMPTY':
                    try:
                        geom = wkt.loads(pix_wkt)
                        if hasattr(geom, 'exterior') and geom.exterior:
                            coords = [(float(pt[0]), float(pt[1])) for pt in geom.exterior.coords]
                            if i_id not in image_polygons:
                                image_polygons[i_id] = []
                            image_polygons[i_id].append(coords)
                    except Exception:
                        continue

        for i_id, polys in image_polygons.items():
            if len(polys) >= 5:
                img_id = i_id
                polygons = polys
                is_spacenet_gt = True
                debug_info = {
                    'matching_rows': len(polys),
                    'non_empty_count': len(polys),
                    'parsed_count': len(polys)
                }
                break

    img_path = None
    for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
        cand = os.path.join(default_img_dir, f"RGB-PanSharpen_{img_id}{ext}")
        if os.path.exists(cand):
            img_path = cand
            break

    if img_path is None and os.path.exists(default_img_dir):
        files = [os.path.join(default_img_dir, f) for f in os.listdir(default_img_dir) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
        if files:
            img_path = files[0]
            img_id = extract_image_id_from_path(img_path)
            if valid_csv_path:
                polygons, matching_rows, non_empty_count, parsed_count = debug_and_load_polygons_from_csv(valid_csv_path, img_id)
                debug_info = {
                    'matching_rows': matching_rows,
                    'non_empty_count': non_empty_count,
                    'parsed_count': parsed_count
                }
                if polygons:
                    is_spacenet_gt = True

    if img_path is None or not os.path.exists(img_path):
        raise FileNotFoundError("No satellite image file found. Please select a valid .tif or .png image.")

    image = load_image_file(img_path)
    if not polygons:
        ext = os.path.splitext(img_path)[1].lower()
        gsd = 0.30 if ext in ['.tif', '.tiff'] else 0.50
        polygons = detect_buildings_from_aerial_image(image, gsd=gsd)

    return image, polygons, img_path, img_id, is_spacenet_gt, debug_info


def estimate_shadow_length_and_height(
    gray_image,
    polygon_coords,
    gsd_meters_per_pixel=0.30,
    solar_elevation_deg=52.0,
    solar_azimuth_deg=135.0,
    floor_height_m=3.0
):
    """
    Robust shadow-based height calculator.
    Measures dark connected shadow regions adjacent to the sun-facing building perimeter.
    """
    if gray_image is None or len(polygon_coords) < 3:
        return {
            'shadow_px': 0.0,
            'shadow_m': 0.0,
            'height_m': 6.0,
            'floors': 2,
            'shadow_valid': False,
            'source': 'Nominal Estimate (No shadow detected)'
        }

    h_img, w_img = gray_image.shape[:2]

    rad_az = math.radians(solar_azimuth_deg)
    dx_shadow = -math.sin(rad_az)
    dy_shadow = math.cos(rad_az)

    pts = np.array(polygon_coords, dtype=np.int32)
    roof_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(roof_mask, [pts], 255)

    roof_pixels = gray_image[roof_mask == 255]
    if len(roof_pixels) == 0:
        return {
            'shadow_px': 0.0,
            'shadow_m': 0.0,
            'height_m': 6.0,
            'floors': 2,
            'shadow_valid': False,
            'source': 'Nominal Estimate (No roof pixels)'
        }

    mean_roof_val = float(np.mean(roof_pixels))
    shadow_threshold = min(mean_roof_val * 0.65, 70.0)

    ray_lengths = []
    max_search_px = 50

    step_stride = max(1, len(pts) // 16)
    for pt in pts[::step_stride]:
        x0, y0 = float(pt[0]), float(pt[1])
        length = 0
        for step in range(1, max_search_px):
            sx = int(round(x0 + dx_shadow * step))
            sy = int(round(y0 + dy_shadow * step))
            if 0 <= sx < w_img and 0 <= sy < h_img:
                if roof_mask[sy, sx] == 255:
                    continue
                val = gray_image[sy, sx]
                if val < shadow_threshold:
                    length = step
                else:
                    if length > 0:
                        break
            else:
                break
        if length >= 3:
            ray_lengths.append(length)

    if len(ray_lengths) >= 3:
        shadow_px = float(np.median(ray_lengths))
        shadow_m = shadow_px * gsd_meters_per_pixel
        elevation_rad = math.radians(solar_elevation_deg)
        height_m = shadow_m * math.tan(elevation_rad)
        height_m = max(2.5, min(height_m, 60.0))
        floors = max(1, int(round(height_m / floor_height_m)))
        return {
            'shadow_px': shadow_px,
            'shadow_m': shadow_m,
            'height_m': height_m,
            'floors': floors,
            'shadow_valid': True,
            'source': 'Shadow Measurement'
        }

    return {
        'shadow_px': 0.0,
        'shadow_m': 0.0,
        'height_m': 6.0,
        'floors': 2,
        'shadow_valid': False,
        'source': 'Nominal Estimate (No shadow detected)'
    }


def calculate_building_dimensions(
    polygon_coords,
    gray_image=None,
    gsd_meters_per_pixel=0.30,
    solar_elevation_deg=52.0,
    solar_azimuth_deg=135.0,
    floor_height_m=3.0,
    is_spacenet_gt=False
):
    """
    Step 8: Computes 3D building dimensions without dropping small ground-truth polygons.
    """
    pts = np.array(polygon_coords, dtype=np.float32).reshape((-1, 1, 2))

    rect = cv2.minAreaRect(pts)
    center, (side1, side2), angle = rect

    length_px = max(side1, side2)
    width_px = min(side1, side2)

    length_m = length_px * gsd_meters_per_pixel
    width_m = width_px * gsd_meters_per_pixel

    polygon = Polygon(polygon_coords)
    area_px = polygon.area
    area_m = area_px * (gsd_meters_per_pixel ** 2)

    box_pts = cv2.boxPoints(rect)
    box_pts = np.int32(box_pts)

    shadow_info = estimate_shadow_length_and_height(
        gray_image,
        polygon_coords,
        gsd_meters_per_pixel=gsd_meters_per_pixel,
        solar_elevation_deg=solar_elevation_deg,
        solar_azimuth_deg=solar_azimuth_deg,
        floor_height_m=floor_height_m
    )

    height_m = shadow_info['height_m']
    height_px = height_m / gsd_meters_per_pixel
    floors = shadow_info['floors']
    volume_m3 = area_m * height_m

    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    solidity = area_px / max(1.0, hull_area)

    if is_spacenet_gt:
        confidence = "High" if shadow_info['shadow_valid'] else ("High" if area_m >= 30.0 else "Medium")
    else:
        if shadow_info['shadow_valid'] and solidity >= 0.75:
            confidence = "High"
        elif solidity >= 0.70:
            confidence = "Medium"
        else:
            confidence = "Low"

    return {
        'rect': rect,
        'box_points': box_pts,
        'center': (int(center[0]), int(center[1])),
        'length_px': length_px,
        'width_px': width_px,
        'length_m': length_m,
        'width_m': width_m,
        'height_px': height_px,
        'height_m': height_m,
        'floors': floors,
        'volume_m3': volume_m3,
        'shadow_px': shadow_info['shadow_px'],
        'shadow_m': shadow_info['shadow_m'],
        'shadow_valid': shadow_info['shadow_valid'],
        'height_source': shadow_info['source'],
        'confidence': confidence,
        'area_px': area_px,
        'area_m': area_m,
        'angle': angle
    }


def draw_building_measurements(
    image,
    polygon_coords,
    measurement_info,
    building_index=1,
    draw_3d_wireframe=True
):
    """
    Renders 2D ground footprint, bounding box, 3D wireframe extrusions, and text labels.
    """
    annotated = image.copy()

    # Ground Footprint Polygon (Green)
    poly_pts = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Oriented Bounding Box (Cyan)
    box_pts = measurement_info['box_points']
    cv2.polylines(annotated, [box_pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # 3D Wireframe Extrusion (Magenta & Hot Pink)
    if draw_3d_wireframe:
        height_px = measurement_info['height_px']
        offset_x = int(round(-0.45 * height_px))
        offset_y = int(round(-0.65 * height_px))

        top_box_pts = box_pts + np.array([offset_x, offset_y], dtype=np.int32)

        for pt_ground, pt_top in zip(box_pts, top_box_pts):
            cv2.line(
                annotated,
                tuple(pt_ground),
                tuple(pt_top),
                color=(255, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )

        cv2.polylines(
            annotated,
            [top_box_pts],
            isClosed=True,
            color=(255, 105, 180),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    # Text Card
    l_m = measurement_info['length_m']
    w_m = measurement_info['width_m']
    h_m = measurement_info['height_m']
    fl = measurement_info['floors']
    conf = measurement_info['confidence']

    label_line1 = f"B{building_index}: {l_m:.1f}m x {w_m:.1f}m x {h_m:.1f}m H ({fl}Fl)"
    label_line2 = f"Conf: {conf}"

    cx, cy = measurement_info['center']
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.40
    thickness = 1

    (text_w1, text_h1), _ = cv2.getTextSize(label_line1, font, font_scale, thickness)
    (text_w2, text_h2), _ = cv2.getTextSize(label_line2, font, font_scale, thickness)

    card_w = max(text_w1, text_w2) + 8
    card_h = text_h1 + text_h2 + 10

    text_x = max(5, cx - card_w // 2)
    text_y = max(card_h + 5, cy - 10)

    cv2.rectangle(
        annotated,
        (text_x - 3, text_y - card_h + 2),
        (text_x + card_w, text_y + 4),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated,
        label_line1,
        (text_x, text_y - text_h2 - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA
    )

    conf_color = (0, 255, 0) if conf == "High" else ((0, 255, 255) if conf == "Medium" else (0, 165, 255))
    cv2.putText(
        annotated,
        label_line2,
        (text_x, text_y),
        font,
        font_scale,
        conf_color,
        thickness,
        lineType=cv2.LINE_AA
    )

    return annotated


def save_image(path, img):
    """
    Saves an image to disk cleanly handling Unicode paths.
    """
    ext = os.path.splitext(path)[1]
    success, encoded_img = cv2.imencode(ext, img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded_img.tobytes())
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="SpaceNet 2 Interactive 3D Building Measurement System")
    parser.add_argument("--image",  type=str,   default=None,  help="Path to satellite image (.tif, .png, .jpg)")
    parser.add_argument("--no-gui", action="store_true",       help="Disable GUI file selection dialog")
    parser.add_argument("--gsd",    type=float, default=None,  help="Ground sample distance in metres/pixel.")
    args = parser.parse_args()

    start_total_time = time.perf_counter()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    base_dir = os.path.join(script_dir, r"dataset\AOI_3_Paris_Train\AOI_3_Paris_Train")
    csv_path = os.path.join(base_dir, r"summaryData\AOI_3_Paris_Train_Building_Solutions.csv")
    img_dir = os.path.join(base_dir, r"RGB-PanSharpen")

    selected_img_path = args.image

    if selected_img_path is None and not args.no_gui and sys.stdin.isatty():
        print("Opening Windows Explorer file dialog to select a satellite image...")
        selected_img_path = select_image_file_dialog(initial_dir=img_dir if os.path.exists(img_dir) else script_dir)
        if selected_img_path:
            print(f"[User Selected Image]: {selected_img_path}")

    original_sat_path = os.path.join(images_dir, "original_satellite_image.png")
    input_path = os.path.join(images_dir, "measured_building_input.png")
    output_path = os.path.join(images_dir, "measured_building_output.png")
    plot_path = os.path.join(images_dir, "building_measurement_plot.png")
    debug_gt_path = os.path.join(images_dir, "debug_all_spacenet_polygons.png")

    print("==================================================")
    print("SpaceNet 2 Dynamic 3D Building Measurement")
    print(" (Length, Width, Height, Floors & Volume)")
    print("==================================================")

    t0 = time.perf_counter()
    image, polygons, img_path, img_id, is_spacenet_gt, debug_info = load_building_data(
        csv_path, img_dir, selected_img_path=selected_img_path
    )
    t_load = (time.perf_counter() - t0) * 1000

    img_h, img_w = image.shape[:2]

    # Print Detailed SpaceNet Debug Output (Steps 1, 3, 7, 10)
    print("\n==================================================")
    print("SPACE NET GROUND TRUTH DEBUG")
    print("==================================================")
    print(f"Selected image           : {os.path.basename(img_path)}")
    print(f"Extracted ImageId        : {img_id}")
    print(f"Image dimensions         : {img_w} x {img_h}")
    print(f"Image coordinate bounds  : X: 0 -> {img_w}, Y: 0 -> {img_h}")

    if is_spacenet_gt:
        print(f"Matching CSV rows        : {debug_info.get('matching_rows', len(polygons))}")
        print(f"Non-empty polygons       : {debug_info.get('non_empty_count', len(polygons))}")
        print(f"Parsed polygons          : {debug_info.get('parsed_count', len(polygons))}")
        print("\nGround-Truth Polygon Coordinates:")
        for idx, poly in enumerate(polygons, 1):
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            print(f"  GT-{idx}: {len(poly)} vertices | X: {min(xs):.1f} -> {max(xs):.1f} | Y: {min(ys):.1f} -> {max(ys):.1f}")

        # Save diagnostic image overlaying ALL CSV polygons
        generate_diagnostic_gt_image(image, polygons, debug_gt_path, img_id)

    print(f"\n[Detection Mode]        : {'SpaceNet Ground Truth' if is_spacenet_gt else 'Computer Vision Fallback'}")
    print(f"[Buildings Loaded]      : {len(polygons)}")

    save_image(original_sat_path, image)
    save_image(input_path, image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ext_lower = os.path.splitext(img_path)[1].lower()
    if args.gsd is not None:
        gsd = args.gsd
        print(f"[GSD]: Using user-specified {gsd:.4f} m/px")
    elif ext_lower in ('.tif', '.tiff'):
        gsd = 0.30
        print(f"[GSD]: GeoTIFF image detected — using native {gsd:.2f} m/px (SpaceNet 2 standard)")
    else:
        mp = (img_h * img_w) / 1_000_000.0
        gsd = 1.20 if mp >= 8 else (0.80 if mp >= 4 else (0.50 if mp >= 1 else 0.30))
        print(f"[GSD]: Aerial image ({img_h}x{img_w}, {mp:.1f} MP) — auto-estimated {gsd:.2f} m/px.")

    footprints_img = image.copy()
    for poly in polygons:
        poly_pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(footprints_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    annotated_img = image.copy()
    building_metrics = []

    print("\n--- Building 3D Measurement Results ---")
    t_calc_start = time.perf_counter()

    for idx, poly in enumerate(polygons, 1):
        dim = calculate_building_dimensions(
            poly,
            gray_image=gray_image,
            gsd_meters_per_pixel=gsd,
            solar_elevation_deg=52.0,
            solar_azimuth_deg=135.0,
            floor_height_m=3.0,
            is_spacenet_gt=is_spacenet_gt
        )

        building_metrics.append(dim)

        print(f"Building #{idx}:")
        print(f"  Confidence : {dim['confidence']} ({dim['height_source']})")
        print(f"  Footprint  : {dim['area_px']:.2f} px² -> {dim['area_m']:.2f} m² (actual polygon area)")
        print(f"  Length     : {dim['length_px']:.2f} px  -> {dim['length_m']:.2f} m (oriented bbox length)")
        print(f"  Width      : {dim['width_px']:.2f} px  -> {dim['width_m']:.2f} m (oriented bbox width)")
        print(f"  Height     : {dim['height_px']:.2f} px  -> {dim['height_m']:.2f} m ({dim['floors']} Floors)")
        print(f"  3D Volume  : {dim['volume_m3']:.2f} m³")
        print("  " + "-" * 45)

        annotated_img = draw_building_measurements(
            annotated_img,
            poly,
            dim,
            building_index=idx,
            draw_3d_wireframe=True
        )

    t_calc_ms = (time.perf_counter() - t_calc_start) * 1000
    print(f"\n[Buildings Rendered]    : {len(building_metrics)}")
    print(f"[Buildings Measured]    : {len(building_metrics)}")
    print(f"[Execution Speed]       : {t_calc_ms:.1f}ms total ({t_calc_ms / max(1, len(building_metrics)):.2f}ms/building)")

    save_image(output_path, annotated_img)
    print(f"[Saved Annotated Output Image]: {output_path}")

    # 4-Panel Plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"1. Original Satellite Image\n({img_id})", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(footprints_img, cv2.COLOR_BGR2RGB))
    footprint_source_str = "SpaceNet Ground-Truth CSV" if is_spacenet_gt else "Computer Vision Fallback"
    axes[0, 1].set_title(f"2. Building Ground Footprints ({len(polygons)})\nSource: {footprint_source_str}", fontsize=12, fontweight='bold')
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"3. 3D Measurement (L x W x H + Extrusions)\n(GSD: {gsd:.2f}m/px, 1 Fl = 3.0m)", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")

    indices = np.arange(1, len(building_metrics) + 1)
    heights = [d['height_m'] for d in building_metrics]
    volumes = [d['volume_m3'] for d in building_metrics]

    ax_bar = axes[1, 1]
    color_h = '#1f77b4'
    color_v = '#ff7f0e'

    if len(building_metrics) > 0:
        ax_bar.bar(indices - 0.2, heights, width=0.4, color=color_h, align='center', label='Height (m)')
        ax_bar.set_xlabel('Building Index (#)', fontsize=11, fontweight='bold')
        ax_bar.set_ylabel('Height (meters)', color=color_h, fontsize=11, fontweight='bold')
        ax_bar.tick_params(axis='y', labelcolor=color_h)
        ax_bar.set_xticks(indices)
        ax_bar.grid(True, linestyle='--', alpha=0.5)

        ax_vol = ax_bar.twinx()
        ax_vol.plot(indices, volumes, color=color_v, marker='o', linewidth=2, label='Volume (m³)')
        ax_vol.set_ylabel('3D Volume (m³)', color=color_v, fontsize=11, fontweight='bold')
        ax_vol.tick_params(axis='y', labelcolor=color_v)
        ax_bar.set_title("4. Building Heights & 3D Volume Distribution", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"[Saved Multi-Panel Analytics Plot]: {plot_path}")

    total_time_ms = (time.perf_counter() - start_total_time) * 1000
    print(f"\nExecution completed successfully in {total_time_ms:.1f}ms!")


if __name__ == '__main__':
    main()
