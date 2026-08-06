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
        root.withdraw()  # Hide root window
        root.attributes('-topmost', True)  # Bring dialog to front

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

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 3:
        if ext in ['.tif', '.tiff']:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def extract_image_id_from_path(img_path):
    """
    Extracts SpaceNet ImageId (e.g. 'AOI_3_Paris_img123') from filename.
    """
    basename = os.path.splitext(os.path.basename(img_path))[0]
    if basename.startswith("RGB-PanSharpen_"):
        return basename.replace("RGB-PanSharpen_", "")
    return basename


def load_polygons_from_csv(csv_path, target_img_id):
    """
    Finds building footprint polygons for target_img_id inside CSV file.
    """
    if not os.path.exists(csv_path):
        return []

    polygons = []
    with open(csv_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['ImageId'] == target_img_id:
                pix_wkt = row['PolygonWKT_Pix']
                if pix_wkt != 'POLYGON EMPTY':
                    geom = wkt.loads(pix_wkt)
                    coords = [(pt[0], pt[1]) for pt in geom.exterior.coords]
                    polygons.append(coords)
    return polygons


def _is_binary_mask(image):
    """
    Heuristic: returns True when the image looks like a binary segmentation mask
    (only two intensity clusters: near-0 and near-255).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    dark = float(np.sum(hist[:30]))
    bright = float(np.sum(hist[225:]))
    total = float(gray.size)
    return (dark + bright) / total > 0.80


def extract_building_polygons_from_png_mask(image, min_area_px=500):
    """
    Legacy entry-point — routes to the correct detector based on image type.
    Binary segmentation masks use the fast threshold path; real aerial/satellite
    photos (JPG, RGB GeoTIFF) use the full multi-strategy CV pipeline.
    """
    if _is_binary_mask(image):
        # ---- Binary-mask fast path (original behaviour) -----------------------
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= 15:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                if len(coords) >= 3:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    polygons.append(coords)
        return polygons

    # ---- Real aerial / satellite photo path -----------------------------------
    return detect_buildings_from_aerial_image(image, min_area_px=min_area_px)


def detect_buildings_from_aerial_image(image, min_area_px=None):
    """
    Multi-strategy, image-size-adaptive building detector for real aerial /
    satellite photographs (JPG, RGB GeoTIFF).

    Pipeline:
      A. CLAHE-enhanced Otsu threshold   — primary detection
      B. Adaptive threshold              — catches low-contrast rooftops
      C. Canny edges (scale-adaptive)    — outlines for large buildings
      D. HSV suppression of sky/veg      — removes false positives
      Merge A+B+C, suppress D, clean up with morphology, then separate
      touching blobs via watershed.  Filter by area, aspect, solidity.
    """
    if image is None:
        return []

    img_h, img_w = image.shape[:2]
    img_area   = img_h * img_w
    short_side = min(img_h, img_w)

    # ------------------------------------------------------------------ #
    # Scale-adaptive morphological kernel sizes (proportional to image)
    #   k_small : ~1 % of short side, minimum 3
    #   k_close : ~3 % for edge-closing, max 21 on large images
    # ------------------------------------------------------------------ #
    k_small = max(3, int(short_side * 0.010) | 1)   # ensure odd
    k_close = max(3, int(short_side * 0.025) | 1)
    k_fill  = max(3, int(short_side * 0.015) | 1)

    # Adaptive min / max area bounds (scale with image resolution)
    if min_area_px is None:
        min_area_px = max(50, int(img_area * 0.0006))   # ≥0.06 % of image
    max_area_px = int(img_area * 0.55)                   # ≤55 % of image

    # ------------------------------------------------------------------ #
    # 0. Preprocessing
    # ------------------------------------------------------------------ #
    bgr = image.copy()
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    blur_k = max(3, (k_small // 2) * 2 - 1)    # odd ≤ k_small
    blurred = cv2.GaussianBlur(bgr, (blur_k, blur_k), 0.8)
    gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    tile = max(4, short_side // 32)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    enhanced = clahe.apply(gray)

    hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]

    # ------------------------------------------------------------------ #
    # Strategy A — Otsu on CLAHE-enhanced greyscale
    # ------------------------------------------------------------------ #
    _, mask_a = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ------------------------------------------------------------------ #
    # Strategy B — Adaptive threshold (local mean) for low-contrast areas
    # ------------------------------------------------------------------ #
    block = max(11, (short_side // 12) | 1)   # odd, scales with image
    mask_b = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block, 4)

    # ------------------------------------------------------------------ #
    # Strategy C — Canny edges closed into blobs (scale-adaptive kernel)
    # ------------------------------------------------------------------ #
    med_val = float(np.median(enhanced))
    canny_lo = int(max(10, med_val * 0.40))
    canny_hi = int(min(240, med_val * 1.20))
    edges  = cv2.Canny(enhanced, canny_lo, canny_hi)
    close_el = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    mask_c   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_el)

    # ------------------------------------------------------------------ #
    # Strategy D — HSV suppression: vegetation, sky, deep shadows
    # ------------------------------------------------------------------ #
    veg_mask    = cv2.inRange(hsv, np.array([30, 35, 35]),  np.array([85,  255, 255]))
    sky_mask    = cv2.inRange(hsv, np.array([90, 15, 150]), np.array([130, 255, 255]))
    shadow_mask = (v_ch < 35).astype(np.uint8) * 255

    suppress = cv2.bitwise_or(veg_mask, sky_mask)
    suppress = cv2.bitwise_or(suppress, shadow_mask)
    sup_el   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    suppress = cv2.dilate(suppress, sup_el)

    # ------------------------------------------------------------------ #
    # Merge A + B + C, then suppress D
    # ------------------------------------------------------------------ #
    combined = cv2.bitwise_or(mask_a, mask_b)
    combined = cv2.bitwise_or(combined, mask_c)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(suppress))

    # ------------------------------------------------------------------ #
    # Morphological cleanup
    # ------------------------------------------------------------------ #
    fill_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fill, k_fill))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, fill_el)
    open_el  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  open_el)

    # ------------------------------------------------------------------ #
    # Watershed separation of touching blobs
    # ------------------------------------------------------------------ #
    dist      = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    fg_thresh = 0.35   # pixels clearly inside a building
    _, sure_fg = cv2.threshold(dist_norm, fg_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg    = (sure_fg * 255).astype(np.uint8)

    sure_bg = cv2.dilate(combined,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                         iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ws_input = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_input, markers)

    # ------------------------------------------------------------------ #
    # Helper: validate & convert a single contour to polygon coords
    # ------------------------------------------------------------------ #
    def _contour_to_polygon(cnt):
        area = cv2.contourArea(cnt)
        if area < min_area_px or area > max_area_px:
            return None
        rect   = cv2.minAreaRect(cnt)
        s1, s2 = rect[1]
        if min(s1, s2) < 2:
            return None
        if max(s1, s2) / min(s1, s2) > 25.0:
            return None
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1 or area / hull_area < 0.28:
            return None
        eps    = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        coords = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return coords

    # ------------------------------------------------------------------ #
    # Extract one polygon per watershed label
    # ------------------------------------------------------------------ #
    polygons = []
    for label in range(2, n_labels + 1):
        lmask = np.where(markers == label, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(lmask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            poly = _contour_to_polygon(cnt)
            if poly:
                polygons.append(poly)

    # ------------------------------------------------------------------ #
    # Fallback: direct contours from combined mask (no watershed)
    # ------------------------------------------------------------------ #
    if not polygons:
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            poly = _contour_to_polygon(cnt)
            if poly:
                polygons.append(poly)

    # ------------------------------------------------------------------ #
    # Additional pass: also try contours on the raw Otsu mask alone
    # (catches buildings missed when combined mask over-merged regions)
    # ------------------------------------------------------------------ #
    seen_centers = set()
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        seen_centers.add((cx // 10, cy // 10))

    fill_a = cv2.morphologyEx(mask_a, cv2.MORPH_CLOSE, fill_el)
    fill_a = cv2.morphologyEx(fill_a, cv2.MORPH_OPEN,  open_el)
    extra_cnts, _ = cv2.findContours(fill_a, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    for cnt in extra_cnts:
        poly = _contour_to_polygon(cnt)
        if poly:
            pts = np.array(poly, dtype=np.float32)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            key = (cx // 10, cy // 10)
            if key not in seen_centers:
                seen_centers.add(key)
                polygons.append(poly)

    print(f"[Detector]: Found {len(polygons)} building regions "
          f"(min={min_area_px} px², max={max_area_px} px²)")
    return polygons


def load_building_data(csv_path, default_img_dir, selected_img_path=None):
    """
    Dynamically loads image and building footprints based on user selection or defaults.
    """
    image = None
    polygons = []
    img_id = "Custom_Image"

    if selected_img_path and os.path.exists(selected_img_path):
        img_path = selected_img_path
        img_id = extract_image_id_from_path(img_path)
        image = load_image_file(img_path)
        
        # Try loading WKT footprints from CSV
        polygons = load_polygons_from_csv(csv_path, img_id)
        if not polygons:
            # Fallback to contour extraction if CSV entry not found
            print(f"[Notice]: No CSV footprint entry found for ID '{img_id}'. Extracting contours automatically...")
            polygons = extract_building_polygons_from_png_mask(image)

        return image, polygons, img_path, img_id

    # Fallback default: find an image in default_img_dir with at least 5 buildings
    if os.path.exists(csv_path):
        image_polygons = {}
        with open(csv_path, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                i_id = row['ImageId']
                pix_wkt = row['PolygonWKT_Pix']
                if pix_wkt != 'POLYGON EMPTY':
                    geom = wkt.loads(pix_wkt)
                    coords = [(pt[0], pt[1]) for pt in geom.exterior.coords]
                    if i_id not in image_polygons:
                        image_polygons[i_id] = []
                    image_polygons[i_id].append(coords)

        for i_id, polys in image_polygons.items():
            if len(polys) >= 5:
                img_id = i_id
                polygons = polys
                break

    # Look for candidate file matching default img_id
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
            polygons = load_polygons_from_csv(csv_path, img_id)

    if img_path is None or not os.path.exists(img_path):
        raise FileNotFoundError("No satellite image file found. Please select a valid .tif or .png image.")

    image = load_image_file(img_path)
    if not polygons:
        polygons = extract_building_polygons_from_png_mask(image)

    return image, polygons, img_path, img_id


def estimate_shadow_length(gray_image, polygon_coords, solar_azimuth_deg=135.0, max_search_px=50):
    """
    Highly optimized shadow length estimator using localized sub-image cropping
    and geospatial solar vector math.
    """
    if gray_image is None or len(polygon_coords) < 3:
        return 0.0

    h_img, w_img = gray_image.shape[:2]

    # Geospatial Solar Vector Math (Azimuth measured clockwise from North)
    rad = math.radians(solar_azimuth_deg)
    dx_shadow = -math.sin(rad)
    dy_shadow = math.cos(rad)

    pts_arr = np.array(polygon_coords, dtype=np.float32)
    min_x = max(0, int(np.min(pts_arr[:, 0])) - max_search_px)
    max_x = min(w_img, int(np.max(pts_arr[:, 0])) + max_search_px)
    min_y = max(0, int(np.min(pts_arr[:, 1])) - max_search_px)
    max_y = min(h_img, int(np.max(pts_arr[:, 1])) + max_search_px)

    crop_gray = gray_image[min_y:max_y, min_x:max_x]
    crop_h, crop_w = crop_gray.shape[:2]

    if crop_h <= 0 or crop_w <= 0:
        return 0.0

    local_pts = pts_arr - np.array([min_x, min_y], dtype=np.float32)
    int_local_pts = np.int32(local_pts)

    local_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(local_mask, [int_local_pts], 255)

    roof_pixels = crop_gray[local_mask == 255]
    if len(roof_pixels) == 0:
        return 0.0

    mean_roof_val = np.mean(roof_pixels)
    shadow_threshold = min(mean_roof_val * 0.72, 105.0)

    shadow_lengths = []
    step_stride = max(1, len(local_pts) // 16)
    for pt in local_pts[::step_stride]:
        x0, y0 = pt[0], pt[1]
        length = 0
        for step in range(1, max_search_px):
            sx = int(round(x0 + dx_shadow * step))
            sy = int(round(y0 + dy_shadow * step))
            if 0 <= sx < crop_w and 0 <= sy < crop_h:
                if local_mask[sy, sx] == 255:
                    continue
                val = crop_gray[sy, sx]
                if val < shadow_threshold:
                    length = step
                else:
                    if length > 0:
                        break
            else:
                break
        if length > 0:
            shadow_lengths.append(length)

    if shadow_lengths:
        return float(np.percentile(shadow_lengths, 75))
    return 0.0


def calculate_building_dimensions(
    polygon_coords, 
    gray_image=None,
    gsd_meters_per_pixel=0.30, 
    solar_elevation_deg=52.0, 
    solar_azimuth_deg=135.0,
    floor_height_m=3.0
):
    """
    Computes accurate 3D building dimensions: Length, Width, Height, Floor Count, and Volume.
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

    shadow_px = estimate_shadow_length(
        gray_image, 
        polygon_coords, 
        solar_azimuth_deg=solar_azimuth_deg
    )
    shadow_m = shadow_px * gsd_meters_per_pixel
    elevation_rad = math.radians(solar_elevation_deg)
    height_shadow_m = shadow_m * math.tan(elevation_rad) if shadow_px > 0 else 0.0

    aspect_ratio = length_m / max(width_m, 0.1)
    base_floors = max(1.0, 0.82 * math.pow(area_m, 0.33) + 0.25 * aspect_ratio)
    height_structural_m = base_floors * floor_height_m

    if height_shadow_m > 2.0:
        height_m = 0.60 * height_shadow_m + 0.40 * height_structural_m
    else:
        height_m = height_structural_m

    height_px = height_m / gsd_meters_per_pixel
    floors = max(1, int(round(height_m / floor_height_m)))
    volume_m3 = area_m * height_m

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
        'shadow_px': shadow_px,
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
    Renders 2D footprint, oriented bounding box, 3D wireframe extrusions, and text cards.
    """
    annotated = image.copy()
    
    # 2D Ground Footprint Polygon (Green)
    poly_pts = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Minimum Area Bounding Box (Cyan)
    box_pts = measurement_info['box_points']
    cv2.polylines(annotated, [box_pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # 3D Wireframe Extrusion Projection (Magenta & Hot Pink)
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

    l_m = measurement_info['length_m']
    w_m = measurement_info['width_m']
    h_m = measurement_info['height_m']
    fl = measurement_info['floors']

    label_str = f"B{building_index}: {l_m:.1f}m x {w_m:.1f}m x {h_m:.1f}m H ({fl}Fl)"

    cx, cy = measurement_info['center']
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)
    
    text_x = max(5, cx - text_w // 2)
    text_y = max(18, cy - 8)

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
    parser.add_argument("--gsd",    type=float, default=None,
                        help="Ground sample distance in metres/pixel. "
                             "Default: 0.30 for TIFF, 0.50 auto-estimated for JPG/PNG.")
    args = parser.parse_args()

    start_total_time = time.perf_counter()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    base_dir = os.path.join(script_dir, r"dataset\AOI_3_Paris_Train\AOI_3_Paris_Train")
    csv_path = os.path.join(base_dir, r"summaryData\AOI_3_Paris_Train_Building_Solutions.csv")
    img_dir = os.path.join(base_dir, r"RGB-PanSharpen")
    
    selected_img_path = args.image

    # Open Windows File Dialog if no direct image argument is provided and GUI is enabled
    if selected_img_path is None and not args.no_gui and sys.stdin.isatty():
        print("Opening Windows Explorer file dialog to select a satellite image...")
        selected_img_path = select_image_file_dialog(initial_dir=img_dir if os.path.exists(img_dir) else script_dir)
        if selected_img_path:
            print(f"[User Selected Image]: {selected_img_path}")
        else:
            print("[File Selection Notice]: No file selected from dialog. Loading default sample image...")

    original_sat_path = os.path.join(images_dir, "original_satellite_image.png")
    input_path = os.path.join(images_dir, "measured_building_input.png")
    output_path = os.path.join(images_dir, "measured_building_output.png")
    plot_path = os.path.join(images_dir, "building_measurement_plot.png")

    print("==================================================")
    print("SpaceNet 2 Dynamic 3D Building Measurement")
    print(" (Length, Width, Height, Floors & Volume)")
    print("==================================================")

    t0 = time.perf_counter()
    image, polygons, img_path, img_id = load_building_data(csv_path, img_dir, selected_img_path=selected_img_path)
    t_load = (time.perf_counter() - t0) * 1000
    print(f"[Loaded Image & Polygons in {t_load:.1f}ms]: {os.path.basename(img_path)} (ID: {img_id})")
    print(f"[Building Count]: {len(polygons)} building footprints found.")

    save_image(original_sat_path, image)
    save_image(input_path, image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- Resolve Ground Sample Distance ----------------------------------------
    ext_lower = os.path.splitext(img_path)[1].lower()
    if args.gsd is not None:
        gsd = args.gsd
        print(f"[GSD]: Using user-specified {gsd:.4f} m/px")
    elif ext_lower in ('.tif', '.tiff'):
        gsd = 0.30          # SpaceNet 2 GeoTIFF native resolution
        print(f"[GSD]: TIFF image detected — using {gsd:.2f} m/px (SpaceNet default)")
    else:
        # Heuristic for aerial JPG: larger images → higher altitude → coarser GSD
        img_h, img_w = image.shape[:2]
        mp = (img_h * img_w) / 1_000_000.0
        if mp >= 8:
            gsd = 1.20
        elif mp >= 4:
            gsd = 0.80
        elif mp >= 1:
            gsd = 0.50
        else:
            gsd = 0.30
        print(f"[GSD]: JPG image ({img_h}x{img_w}, {mp:.1f} MP) — "
              f"auto-estimated {gsd:.2f} m/px. Override with --gsd <value>.")

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
            floor_height_m=3.0
        )
        
        if dim['area_m'] < 1.0:
            continue

        building_metrics.append(dim)

        print(f"Building #{idx}:")
        print(f"  Length   : {dim['length_px']:.2f} px  -> {dim['length_m']:.2f} m")
        print(f"  Width    : {dim['width_px']:.2f} px  -> {dim['width_m']:.2f} m")
        print(f"  Height   : {dim['height_px']:.2f} px  -> {dim['height_m']:.2f} m ({dim['floors']} Floors)")
        print(f"  Footprint: {dim['area_px']:.2f} px² -> {dim['area_m']:.2f} m²")
        print(f"  3D Volume: {dim['volume_m3']:.2f} m³")
        print("  " + "-" * 40)

        annotated_img = draw_building_measurements(
            annotated_img,
            poly,
            dim,
            building_index=idx,
            draw_3d_wireframe=True
        )

    t_calc_ms = (time.perf_counter() - t_calc_start) * 1000
    print(f"\n[Calculated & Rendered {len(building_metrics)} Buildings in {t_calc_ms:.1f}ms] ({t_calc_ms / max(1, len(building_metrics)):.2f}ms/building)")

    save_image(output_path, annotated_img)
    print(f"[Saved Annotated Output Image]: {output_path}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"1. Original Satellite Image\n({img_id})", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(footprints_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"2. Building Ground Footprints\n(Green Polygons)", fontsize=12, fontweight='bold')
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

    bars = ax_bar.bar(indices - 0.2, heights, width=0.4, color=color_h, align='center', label='Height (m)')
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
