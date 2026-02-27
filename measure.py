"""
measure.py – OpenCV grid-based object measurement.

Pipeline:
  1. Detect grid lines (Hough)
  2. Perspective correction (Homography)
  3. Object contour detection
  4. Bounding box → pixel → mm  (based on known grid square size)

Measurement is REFUSED if grid is not detected or calibration check fails.
"""

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("organizer.measure")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum number of grid lines to consider detection successful
MIN_GRID_LINES_H = 4
MIN_GRID_LINES_V = 4

# Calibration: max allowed coefficient of variation for detected square sizes
MAX_SQUARE_SIZE_CV = 0.15  # 15% — rejects badly angled photos

# Minimum object area as fraction of image area (reject noise)
MIN_OBJECT_AREA_FRAC = 0.005

# Edge detection compensation: pixels to add to length measurement
# Compensates for edge detection not reaching the very tips of objects
EDGE_COMPENSATION_PX = 30  # ~15 pixels per end


class MeasurementResult:
    def __init__(
        self,
        width_mm: float,
        depth_mm: float,
        width_px: int,
        depth_px: int,
        grid_detected: bool,
        calibration_ok: bool,
        px_per_mm: float,
        debug_image: Optional[np.ndarray] = None,
        # Extended measurements for complex objects
        length_mm: float = 0.0,  # Length along main axis
        handle_width_mm: float = 0.0,  # Width of thickest part
        shaft_length_mm: float = 0.0,  # Length of thinner part
        angle: float = 0.0,  # Rotation angle of object
    ):
        self.width_mm = round(width_mm, 1)
        self.depth_mm = round(depth_mm, 1)
        self.width_px = width_px
        self.depth_px = depth_px
        self.grid_detected = grid_detected
        self.calibration_ok = calibration_ok
        self.px_per_mm = px_per_mm
        self.debug_image = debug_image
        # Extended measurements in mm
        self.length_mm = round(length_mm, 2)
        self.handle_width_mm = round(handle_width_mm, 2)
        self.shaft_length_mm = round(shaft_length_mm, 2)
        self.angle = round(angle, 1)
        # Also provide values in cm for convenience
        self.width_cm = round(width_mm / 10, 2)
        self.depth_cm = round(depth_mm / 10, 2)
        self.length_cm = round(length_mm / 10, 2)


class MeasurementError(Exception):
    pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def measure_object(image_bytes: bytes, grid_square_mm: int = 20) -> MeasurementResult:
    """
    Measure an object from a top-down photo on a grid background.

    Args:
        image_bytes: JPEG/PNG image data
        grid_square_mm: Known size of each grid square in mm

    Returns:
        MeasurementResult with width and depth in mm

    Raises:
        MeasurementError if grid not detected or calibration fails
    """
    # Decode image
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise MeasurementError("Could not decode image.")

    # Step 1: Detect grid
    corners, h_lines, v_lines = _detect_grid(img)

    if len(h_lines) < MIN_GRID_LINES_H or len(v_lines) < MIN_GRID_LINES_V:
        raise MeasurementError(
            f"Grid not detected: found {len(h_lines)}h/{len(v_lines)}v lines "
            f"(need {MIN_GRID_LINES_H}h/{MIN_GRID_LINES_V}v). "
            "Ensure top-down photo on grid background."
        )

    # Step 2: Perspective correction via homography
    corrected, px_per_square_tuple = _correct_perspective(img, h_lines, v_lines, corners)
    h_px_per_square, v_px_per_square = px_per_square_tuple

    # Step 3: Calibration check — are detected squares consistent in size?
    square_sizes = _measure_square_sizes(h_lines, v_lines, corrected)
    if len(square_sizes) < 4:
        raise MeasurementError("Not enough grid intersections for calibration check.")

    cv_val = _coefficient_of_variation(square_sizes)
    if cv_val > MAX_SQUARE_SIZE_CV:
        raise MeasurementError(
            f"Calibration failed: grid square size variance too high "
            f"(CV={cv_val:.2%}, max={MAX_SQUARE_SIZE_CV:.0%}). "
            "Camera may be tilted. Ensure perpendicular top-down shot."
        )

    # Step 4: Detect object contour and get detailed measurements
    detection_result = _detect_object_with_segments(corrected, img_area_frac=MIN_OBJECT_AREA_FRAC)

    if detection_result is None:
        raise MeasurementError("No object detected on the grid. Ensure object contrasts with background.")

    # Unpack detection results
    bbox, contour, segments = detection_result
    cx, cy, w_px, h_px, angle = bbox
    
    # Use average of both spacings for consistent measurements
    avg_px_per_square = (h_px_per_square + v_px_per_square) / 2
    px_per_mm = avg_px_per_square / grid_square_mm
    
    # Apply edge compensation to length (adds pixels to compensate for edge detection losses)
    w_px_compensated = w_px + EDGE_COMPENSATION_PX
    
    # Primary dimensions (rotated bounding box)
    width_mm = w_px_compensated / px_per_mm
    depth_mm = h_px / px_per_mm
    
    # Calculate extended measurements from segments
    length_mm = width_mm  # Length is the longer dimension
    handle_width_mm = 0.0
    shaft_length_mm = 0.0
    
    if segments:
        # Analyze segments to find handle (wide part) and shaft (thin part)
        handle_seg, shaft_seg = _analyze_segments(segments, px_per_mm)
        if handle_seg:
            handle_width_mm = handle_seg['height_mm']
        if shaft_seg:
            shaft_length_mm = shaft_seg['width_mm']
    
    # Build debug image with detailed measurements
    debug = corrected.copy()
    rect = ((cx, cy), (w_px, h_px), angle)
    box_points = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(debug, [box_points], 0, (0, 255, 0), 3)
    
    # Draw contour for reference
    if contour is not None:
        cv2.drawContours(debug, [contour], 0, (255, 165, 0), 2)
    
    # Add dimension labels
    _draw_measurements_on_image(debug, cx, cy, w_px, h_px, angle, width_mm, depth_mm, segments, px_per_mm)

    log.info("Measured: %.2f x %.2f mm (length=%.2f, handle_w=%.2f, shaft_l=%.2f, angle=%.1f°, px_per_mm=%.2f)",
             width_mm, depth_mm, length_mm, handle_width_mm, shaft_length_mm, angle, px_per_mm)

    return MeasurementResult(
        width_mm=width_mm,
        depth_mm=depth_mm,
        width_px=w_px,
        depth_px=h_px,
        grid_detected=True,
        calibration_ok=True,
        px_per_mm=px_per_mm,
        debug_image=debug,
        length_mm=length_mm,
        handle_width_mm=handle_width_mm,
        shaft_length_mm=shaft_length_mm,
        angle=angle,
    )


# ---------------------------------------------------------------------------
# Step 1: Grid line detection
# ---------------------------------------------------------------------------

def _detect_grid(img: np.ndarray) -> Tuple[np.ndarray, list, list]:
    """
    Detect horizontal and vertical grid lines using Hough transform.
    Returns: (corner_points, h_lines, v_lines)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )

    # Detect lines with HoughLinesP
    lines = cv2.HoughLinesP(
        thresh, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=min(img.shape[:2]) // 6,
        maxLineGap=20
    )

    if lines is None:
        return np.array([]), [], []

    h_lines = []
    v_lines = []
    corners = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

        if angle < 15 or angle > 165:  # Horizontal
            h_lines.append((x1, y1, x2, y2))
        elif 75 < angle < 105:  # Vertical
            v_lines.append((x1, y1, x2, y2))

    # Cluster lines to deduplicate
    h_lines = _cluster_lines(h_lines, axis="h", threshold=15)
    v_lines = _cluster_lines(v_lines, axis="v", threshold=15)

    # Find intersections as potential corners
    for hl in h_lines:
        for vl in v_lines:
            pt = _line_intersection(hl, vl)
            if pt is not None:
                corners.append(pt)

    return np.array(corners) if corners else np.array([]), h_lines, v_lines


def _cluster_lines(lines: list, axis: str, threshold: int = 15) -> list:
    """Merge nearby parallel lines into single representative lines."""
    if not lines:
        return []

    if axis == "h":
        # Sort by y midpoint
        lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        key_fn = lambda l: (l[1] + l[3]) / 2
    else:
        # Sort by x midpoint
        lines.sort(key=lambda l: (l[0] + l[2]) / 2)
        key_fn = lambda l: (l[0] + l[2]) / 2

    clustered = [lines[0]]
    for line in lines[1:]:
        if abs(key_fn(line) - key_fn(clustered[-1])) > threshold:
            clustered.append(line)

    return clustered


def _line_intersection(l1: tuple, l2: tuple) -> Optional[Tuple[int, int]]:
    """Find intersection point of two line segments (extended to infinite lines)."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    return (int(px), int(py))


# ---------------------------------------------------------------------------
# Step 2: Perspective correction
# ---------------------------------------------------------------------------

def _correct_perspective(
    img: np.ndarray,
    h_lines: list,
    v_lines: list,
    corners: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Apply homography correction using detected grid intersections.
    Returns (corrected_image, (h_px_per_square, v_px_per_square)).
    """
    if len(corners) < 4:
        # Not enough corners for homography, estimate from line spacing
        px_per_square = _estimate_px_per_square(h_lines, v_lines)
        return img, px_per_square

    # Sort corners to find the grid bounding rectangle
    # Use convex hull and select 4 outermost corners
    hull = cv2.convexHull(corners.astype(np.float32))
    if len(hull) < 4:
        px_per_square = _estimate_px_per_square(h_lines, v_lines)
        return img, px_per_square

    # Approximate to 4 corners
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) == 4:
        src_pts = approx.reshape(4, 2).astype(np.float32)
    else:
        # Take bounding rect corners from corner points
        rect = cv2.minAreaRect(corners.astype(np.float32))
        src_pts = cv2.boxPoints(rect).astype(np.float32)

    # Order: top-left, top-right, bottom-right, bottom-left
    src_pts = _order_points(src_pts)

    # Calculate target dimensions based on grid spacing
    h_spacing = _estimate_spacing(h_lines, axis="h")
    v_spacing = _estimate_spacing(v_lines, axis="v")

    if h_spacing < 10 and v_spacing < 10:
        # Spacing too small, skip homography
        fallback = max(h_spacing, v_spacing) if max(h_spacing, v_spacing) > 0 else 50
        return img, (fallback, fallback)

    # Destination rectangle
    width = int(np.linalg.norm(src_pts[1] - src_pts[0]))
    height = int(np.linalg.norm(src_pts[3] - src_pts[0]))

    if width < 100 or height < 100:
        return img, (h_spacing if h_spacing > 0 else v_spacing, v_spacing if v_spacing > 0 else h_spacing)

    dst_pts = np.float32([
        [0, 0], [width, 0], [width, height], [0, height]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, M, (width, height))

    return corrected, (h_spacing, v_spacing)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.float32([
        pts[np.argmin(s)],   # top-left
        pts[np.argmin(d)],   # top-right
        pts[np.argmax(s)],   # bottom-right
        pts[np.argmax(d)],   # bottom-left
    ])


def _estimate_spacing(lines: list, axis: str) -> float:
    """Estimate average spacing between parallel lines (in pixels)."""
    if len(lines) < 2:
        return 0

    if axis == "h":
        positions = sorted([(l[1] + l[3]) / 2 for l in lines])
    else:
        positions = sorted([(l[0] + l[2]) / 2 for l in lines])

    spacings = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    return np.median(spacings) if spacings else 0


def _estimate_px_per_square(h_lines: list, v_lines: list) -> Tuple[float, float]:
    """Returns (h_px_per_square, v_px_per_square)."""
    h_sp = _estimate_spacing(h_lines, "h")
    v_sp = _estimate_spacing(v_lines, "v")
    # Return both values; use max or 50 as fallback for missing
    h_val = h_sp if h_sp > 0 else (v_sp if v_sp > 0 else 50)
    v_val = v_sp if v_sp > 0 else (h_sp if h_sp > 0 else 50)
    return h_val, v_val


# ---------------------------------------------------------------------------
# Step 3: Calibration check
# ---------------------------------------------------------------------------

def _measure_square_sizes(h_lines: list, v_lines: list, img: np.ndarray) -> list:
    """Measure individual grid cell sizes to check calibration consistency."""
    h_sp = _estimate_spacing(h_lines, "h")
    v_sp = _estimate_spacing(v_lines, "v")

    sizes = []
    if h_sp > 0:
        h_pos = sorted([(l[1] + l[3]) / 2 for l in h_lines])
        sizes.extend([h_pos[i + 1] - h_pos[i] for i in range(len(h_pos) - 1)])
    if v_sp > 0:
        v_pos = sorted([(l[0] + l[2]) / 2 for l in v_lines])
        sizes.extend([v_pos[i + 1] - v_pos[i] for i in range(len(v_pos) - 1)])

    return sizes


def _coefficient_of_variation(values: list) -> float:
    """CV = std/mean. Lower = more consistent."""
    if not values:
        return 999.0
    arr = np.array(values)
    mean = np.mean(arr)
    if mean < 1e-6:
        return 999.0
    return float(np.std(arr) / mean)


# ---------------------------------------------------------------------------
# Step 4: Object detection
# ---------------------------------------------------------------------------

def _detect_object(img: np.ndarray, img_area_frac: float = 0.005) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the main object on the grid background.
    Uses multiple detection methods for robustness:
    1. Color-based detection (HSV) - for colored objects
    2. Grayscale Otsu's threshold - for dark objects
    3. Edge-based detection - fallback for tricky cases
    Returns bounding box (x, y, w, h) or None.
    """
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * img_area_frac
    log.debug("Detection: img_area=%d, min_area=%.0f (%.2f%%)", img_area, min_area, img_area_frac * 100)

    # Try multiple detection methods and combine results
    masks = []

    # Method 1: Color-based detection (detects non-white/non-black regions)
    color_mask = _detect_by_color(img)
    if color_mask is not None:
        masks.append(color_mask)
        nonzero = cv2.countNonZero(color_mask)
        log.debug("Color mask: %d non-zero pixels (%.2f%%)", nonzero, 100.0 * nonzero / img_area)

    # Method 2: Grayscale Otsu's threshold (original method)
    gray_mask = _detect_by_grayscale(img)
    if gray_mask is not None:
        masks.append(gray_mask)
        nonzero = cv2.countNonZero(gray_mask)
        log.debug("Gray mask: %d non-zero pixels (%.2f%%)", nonzero, 100.0 * nonzero / img_area)

    # Method 3: Edge-based detection
    edge_mask = _detect_by_edges(img)
    if edge_mask is not None:
        masks.append(edge_mask)
        nonzero = cv2.countNonZero(edge_mask)
        log.debug("Edge mask: %d non-zero pixels (%.2f%%)", nonzero, 100.0 * nonzero / img_area)

    if not masks:
        log.warning("No detection masks created")
        return None

    # Combine masks - use OR to capture all potential object regions
    combined = masks[0]
    for mask in masks[1:]:
        combined = cv2.bitwise_or(combined, mask)
    
    log.debug("Combined mask: %d non-zero pixels", cv2.countNonZero(combined))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    log.debug("After morphology: %d non-zero pixels", cv2.countNonZero(cleaned))

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        log.warning("No contours found after morphology")
        return None
    
    log.debug("Found %d contours", len(contours))
    for i, c in enumerate(contours[:5]):  # Log first 5
        area = cv2.contourArea(c)
        log.debug("  Contour %d: area=%d (min=%d, valid=%s)", i, area, min_area, area > min_area)

    # Filter by area, find the largest valid contour
    valid = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid:
        log.warning("No contours passed min_area filter (min=%d)", min_area)
        return None

    # Take the largest contour (the object)
    largest = max(valid, key=cv2.contourArea)
    
    # Use minAreaRect for rotated bounding box (handles diagonal objects)
    rect = cv2.minAreaRect(largest)
    center, (w, h), angle = rect
    
    # Ensure width >= height (normalize orientation)
    if w < h:
        w, h = h, w
        angle = angle + 90 if angle < 0 else angle - 90
    
    log.info("Object detected: rotated_bbox center=(%.0f,%.0f) size=(%.0f,%.0f) angle=%.1f°",
             center[0], center[1], w, h, angle)

    # Return as (center_x, center_y, width, height, angle)
    return (int(center[0]), int(center[1]), int(w), int(h), angle)


def _detect_grid_lines(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect grid lines in the image.
    Grid lines are THIN, REGULAR, BLACK structures that are perfectly
    horizontal or vertical. Uses strict detection to avoid catching
    object parts like metallic shafts.
    """
    try:
        h, w = gray.shape
        
        # Use a stricter threshold - grid lines are BLACK (near 0)
        # This avoids catching gray metallic parts
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Grid lines must be VERY thin (1-2 pixels for the actual line)
        # After edge detection and anti-aliasing they might be 3-5 pixels
        
        # Detect horizontal lines using long narrow kernel
        # Must be perfectly horizontal (height=1 or 2)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        h_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, h_kernel)
        
        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
        v_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, v_kernel)
        
        # Combine horizontal and vertical lines
        grid_mask = cv2.bitwise_or(h_lines, v_lines)
        
        # Only dilate slightly (2-3 pixels each direction)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid_mask = cv2.dilate(grid_mask, dilate_kernel, iterations=1)
        
        log.debug("Grid lines detected: %d pixels", cv2.countNonZero(grid_mask))
        
        return grid_mask
    except Exception as e:
        log.warning("Grid line detection failed: %s", e)
        return None


def _detect_by_color(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect colored objects using HSV color space.
    Grid is white+black (low saturation), objects typically have color.
    Enhanced to detect metallic/gray objects like screwdriver shafts.
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # First, detect and exclude grid lines
        grid_mask = _detect_grid_lines(gray)
        
        # Detect saturated colors (not white, gray, or black)
        # These are the main object colors (e.g., yellow handle)
        lower_sat = np.array([0, 40, 40])   # min hue, sat, val
        upper_sat = np.array([180, 255, 255])  # max hue, sat, val
        color_mask = cv2.inRange(hsv, lower_sat, upper_sat)
        
        # Remove grid lines from color mask
        if grid_mask is not None:
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(grid_mask))
        
        # Remove thin structures but keep solid objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        return color_mask
    except Exception:
        return None


def _detect_by_grayscale(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Grayscale threshold for detecting dark objects.
    Specifically excludes grid lines using morphological operations.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # First, detect and remove grid lines
        # Grid lines are thin, regular, horizontal and vertical structures
        grid_mask = _detect_grid_lines(gray)
        
        # Use Otsu's threshold
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove detected grid lines from the threshold result
        if grid_mask is not None:
            thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(grid_mask))
        
        # Use smaller kernel to preserve thin structures like shafts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opened
    except Exception:
        return None


def _detect_by_edges(img: np.ndarray) -> Optional[np.ndarray]:
    """Edge-based detection using Canny + contour filling.
    Only used as fallback - disabled by default as it picks up grid lines."""
    # Disabled - edge detection picks up grid lines and dominates the result
    # Only enable if color and grayscale both fail
    return None


# ---------------------------------------------------------------------------
# Enhanced detection with segment analysis
# ---------------------------------------------------------------------------

def _detect_metallic(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect metallic/gray objects that might be missed by color detection.
    Metallic objects have medium gray values with low saturation.
    Optimized for thin metallic parts like screwdriver shafts.
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Metallic objects: low saturation (gray), varied brightness
        # Wide range to catch different metallic appearances
        lower_metallic = np.array([0, 0, 80])
        upper_metallic = np.array([180, 60, 220])
        metallic_mask = cv2.inRange(hsv, lower_metallic, upper_metallic)
        
        # Exclude pure white background
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        metallic_mask = cv2.bitwise_and(metallic_mask, cv2.bitwise_not(white_mask))
        
        # For metallic parts, we use a smaller grid exclusion to preserve thin shafts
        # Only remove clearly horizontal/vertical thin structures (grid lines)
        h, w = gray.shape
        
        # Detect strictly horizontal lines (long kernel)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
        
        # Detect strictly vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
        v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
        
        # Combine and dilate slightly
        grid_lines_strict = cv2.bitwise_or(h_lines, v_lines)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid_lines_strict = cv2.dilate(grid_lines_strict, kernel_dilate, iterations=1)
        
        # Remove strict grid lines from metallic mask
        metallic_mask = cv2.bitwise_and(metallic_mask, cv2.bitwise_not(grid_lines_strict))
        
        # Use minimal opening to preserve thin structures
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_OPEN, kernel_small)
        
        return metallic_mask
    except Exception:
        return None


def _detect_thin_structures(img: np.ndarray, colored_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect thin elongated structures that are connected to colored objects.
    This catches thin shafts that might be missed by other methods.
    Uses edge detection and line segment analysis.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Detect edges in the image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find pixels that are gray (not white, not black) - potential shaft regions
        # Shafts are typically medium gray (100-180 range)
        shaft_gray = cv2.inRange(gray, 100, 190)
        
        # Exclude white background
        _, white = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        shaft_gray = cv2.bitwise_and(shaft_gray, cv2.bitwise_not(white))
        
        # Create a mask of regions that are near colored objects (like handles)
        # Dilate the colored mask to create a "search zone"
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        near_colored = cv2.dilate(colored_mask, kernel_dilate, iterations=3)
        
        # Keep only shaft-gray pixels that are near colored objects
        connected_gray = cv2.bitwise_and(shaft_gray, near_colored)
        
        # Remove grid lines using the strict grid detection
        # Grid lines are perfectly horizontal or vertical
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 8))
        
        _, dark = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
        grid_strict = cv2.bitwise_or(h_lines, v_lines)
        
        # Dilate grid mask slightly
        grid_dilated = cv2.dilate(grid_strict, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        
        # Remove strict grid lines from result
        connected_gray = cv2.bitwise_and(connected_gray, cv2.bitwise_not(grid_dilated))
        
        # Clean up
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(connected_gray, cv2.MORPH_OPEN, kernel_open)
        
        return result
        
    except Exception as e:
        log.warning("Thin structure detection failed: %s", e)
        return None


def _detect_by_edge_contours(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect objects using edge detection and contour analysis.
    This method finds closed contours and filters out grid-line contours.
    Works well for objects with distinct boundaries including thin parts.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Apply bilateral filter to preserve edges while smoothing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect edges using Canny
        edges = cv2.Canny(filtered, 30, 100)
        
        # Close small gaps in edges to create closed contours
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Find contours
        contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Create output mask
        result_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Analyze each contour - keep non-grid-like contours
        min_area = (h * w) * 0.003  # Minimum 0.3% of image area
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            center, (rect_w, rect_h), angle = rect
            
            # Skip very thin contours that are aligned with grid (horizontal or vertical)
            if rect_w > 0 and rect_h > 0:
                aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
                
                # Grid lines are perfectly horizontal/vertical and very elongated
                is_horizontal = abs(angle) < 5 or abs(angle) > 175
                is_vertical = abs(angle - 90) < 5 or abs(angle + 90) < 5
                
                # If very elongated AND aligned with grid, it's probably a grid line
                if aspect_ratio > 30 and (is_horizontal or is_vertical):
                    continue
            
            # Keep this contour - fill it in the mask
            cv2.drawContours(result_mask, [contour], 0, 255, -1)
        
        # Clean up small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel_open)
        
        return result_mask
        
    except Exception as e:
        log.warning("Edge contour detection failed: %s", e)
        return None


def _detect_object_with_segments(img: np.ndarray, img_area_frac: float = 0.005) -> Optional[tuple]:
    """
    Detect the main object and analyze its segments.
    
    Returns: (bbox, contour, segments) where:
        - bbox: (cx, cy, width, height, angle) rotated bounding box
        - contour: the object contour
        - segments: list of segment dictionaries with dimensions
    """
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * img_area_frac
    log.debug("Detection: img_area=%d, min_area=%.0f (%.2f%%)", img_area, min_area, img_area_frac * 100)

    masks = []

    # Method 1: Color-based detection
    color_mask = _detect_by_color(img)
    if color_mask is not None:
        masks.append(("color", color_mask))
        log.debug("Color mask: %d non-zero pixels", cv2.countNonZero(color_mask))

    # Method 2: Grayscale Otsu's threshold
    gray_mask = _detect_by_grayscale(img)
    if gray_mask is not None:
        masks.append(("gray", gray_mask))
        log.debug("Gray mask: %d non-zero pixels", cv2.countNonZero(gray_mask))

    # Method 3: Metallic detection (for screwdriver shafts, etc.)
    metallic_mask = _detect_metallic(img)
    if metallic_mask is not None:
        masks.append(("metallic", metallic_mask))
        log.debug("Metallic mask: %d non-zero pixels", cv2.countNonZero(metallic_mask))
    
    # Method 4: Thin structure detection using edges (for thin shafts)
    thin_mask = _detect_thin_structures(img, color_mask if color_mask is not None else np.zeros_like(gray_mask))
    if thin_mask is not None and cv2.countNonZero(thin_mask) > 0:
        masks.append(("thin", thin_mask))
        log.debug("Thin structure mask: %d non-zero pixels", cv2.countNonZero(thin_mask))

    if not masks:
        log.warning("No detection masks created")
        return None

    # Combine all masks
    combined = masks[0][1]
    for name, mask in masks[1:]:
        combined = cv2.bitwise_or(combined, mask)

    log.debug("Combined mask: %d non-zero pixels", cv2.countNonZero(combined))

    # Morphological cleanup
    # Strategy: Close small gaps to connect parts of the same object,
    # then minimal opening to remove only small noise while preserving edges
    
    # Close small gaps within the object (e.g., connect handle to shaft)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Very minimal opening to remove only small noise (3x3 instead of 5x5)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    log.debug("After morphology: %d non-zero pixels", cv2.countNonZero(cleaned))

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        log.warning("No contours found after morphology")
        return None

    # Filter by area
    valid = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid:
        log.warning("No contours passed min_area filter (min=%d)", min_area)
        return None

    # Take the largest contour
    largest = max(valid, key=cv2.contourArea)
    
    # Try to extend the contour to include thin connected parts (like shaft tips)
    # Pass the original image for better shaft detection
    largest = _extend_contour_along_axis(largest, cleaned, img)

    # Compute rotated bounding box
    rect = cv2.minAreaRect(largest)
    center, (w, h), angle = rect

    # Normalize: width >= height
    if w < h:
        w, h = h, w
        angle = angle + 90 if angle < 0 else angle - 90

    bbox = (int(center[0]), int(center[1]), int(w), int(h), angle)

    # Analyze segments along the main axis
    segments = _extract_segments(largest, angle, img.shape)

    log.info("Object detected: bbox=(%d,%d) size=(%d,%d) angle=%.1f° segments=%d",
             center[0], center[1], w, h, angle, len(segments))

    return (bbox, largest, segments)


def _extend_contour_along_axis(contour: np.ndarray, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Extend the contour along its main axis to capture thin connected parts.
    Uses ray casting from the contour endpoints to find additional pixels
    that are part of the object (like metallic shaft tips).
    """
    try:
        img_shape = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        
        # Get the oriented bounding box
        rect = cv2.minAreaRect(contour)
        center, (w, h), angle = rect
        
        # Normalize orientation
        if w < h:
            w, h = h, w
            angle = angle + 90 if angle < 0 else angle - 90
        
        # Calculate direction vectors along the main axis
        angle_rad = math.radians(angle)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        
        # Create extended search corridor along the main axis
        corridor_mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        # Extend up to 80% of current length in each direction
        extension_length = int(w * 0.8)
        corridor_width = max(20, int(h * 0.6))
        
        # Extend from center outward
        ext_p1 = (int(center[0] - (w/2 + extension_length) * dx), 
                  int(center[1] - (w/2 + extension_length) * dy))
        ext_p2 = (int(center[0] + (w/2 + extension_length) * dx), 
                  int(center[1] + (w/2 + extension_length) * dy))
        
        cv2.line(corridor_mask, ext_p1, ext_p2, 255, max(15, corridor_width))
        
        # Detect ALL non-white pixels in the search corridor
        shaft_mask = cv2.inRange(gray, 30, 225)
        
        # Exclude white background
        _, white = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        shaft_mask = cv2.bitwise_and(shaft_mask, cv2.bitwise_not(white))
        
        # Only keep shaft pixels within the search corridor
        shaft_in_corridor = cv2.bitwise_and(shaft_mask, corridor_mask)
        
        # Remove grid lines - only strictly horizontal/vertical thin lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 12, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 12))
        _, very_dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        h_lines = cv2.morphologyEx(very_dark, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(very_dark, cv2.MORPH_OPEN, v_kernel)
        grid_lines = cv2.bitwise_or(h_lines, v_lines)
        grid_lines = cv2.dilate(grid_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        shaft_in_corridor = cv2.bitwise_and(shaft_in_corridor, cv2.bitwise_not(grid_lines))
        
        # Combine with original mask pixels in corridor
        mask_in_corridor = cv2.bitwise_and(mask, corridor_mask)
        additional_pixels = cv2.bitwise_or(shaft_in_corridor, mask_in_corridor)
        
        pixel_count = cv2.countNonZero(additional_pixels)
        
        if pixel_count > 100:
            # Create mask from original contour
            contour_mask = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            
            # Combine with additional pixels
            combined_mask = cv2.bitwise_or(contour_mask, additional_pixels)
            
            # Close gaps - use larger kernel to connect shaft to handle
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
            
            # Small opening to clean noise
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            
            # Find contours in combined mask
            new_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if new_contours:
                new_largest = max(new_contours, key=cv2.contourArea)
                
                new_rect = cv2.minAreaRect(new_largest)
                _, (new_w, new_h), _ = new_rect
                new_length = max(new_w, new_h)
                old_length = max(w, h)
                
                if new_length > old_length * 1.02:
                    log.debug("Extended contour: length %.0f -> %.0f pixels (+%.1f%%)", 
                              old_length, new_length, 100*(new_length-old_length)/old_length)
                    return new_largest
        
        return contour
        
    except Exception as e:
        log.warning("Contour extension failed: %s", e)
        return contour


def _extract_segments(contour: np.ndarray, angle: float, img_shape: tuple) -> list:
    """
    Extract segment information by analyzing cross-sections along the object's main axis.
    Returns list of segments with their positions and widths.
    """
    segments = []
    
    try:
        # Get the oriented bounding box
        rect = cv2.minAreaRect(contour)
        center, (w, h), rect_angle = rect
        
        # Ensure we have the correct orientation
        if w < h:
            w, h = h, w
            rect_angle = rect_angle + 90 if rect_angle < 0 else rect_angle - 90
        
        # Create a mask from the contour
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Sample widths along the main axis
        num_samples = 20
        angle_rad = math.radians(rect_angle)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        
        widths = []
        for i in range(num_samples):
            # Position along the length
            t = (i / (num_samples - 1)) - 0.5  # -0.5 to 0.5
            px = int(center[0] + t * w * dx)
            py = int(center[1] + t * w * dy)
            
            # Measure perpendicular width at this point
            width_px = _measure_perpendicular_width(mask, px, py, rect_angle)
            widths.append({
                'position': t,
                'x': px,
                'y': py,
                'width_px': width_px
            })
        
        # Find segments based on width changes
        # Group similar widths together
        if widths:
            current_segment = {
                'start_pos': widths[0]['position'],
                'widths': [widths[0]['width_px']],
                'positions': [widths[0]['position']]
            }
            
            for w_data in widths[1:]:
                avg_width = np.mean(current_segment['widths'])
                # If width changes significantly (>50%), start new segment
                if avg_width > 0 and abs(w_data['width_px'] - avg_width) / avg_width > 0.5:
                    # Save current segment
                    segments.append({
                        'start_pos': current_segment['start_pos'],
                        'end_pos': current_segment['positions'][-1],
                        'avg_width_px': np.mean(current_segment['widths']),
                        'max_width_px': max(current_segment['widths']),
                        'length_frac': current_segment['positions'][-1] - current_segment['start_pos']
                    })
                    # Start new segment
                    current_segment = {
                        'start_pos': w_data['position'],
                        'widths': [w_data['width_px']],
                        'positions': [w_data['position']]
                    }
                else:
                    current_segment['widths'].append(w_data['width_px'])
                    current_segment['positions'].append(w_data['position'])
            
            # Save last segment
            if current_segment['widths']:
                segments.append({
                    'start_pos': current_segment['start_pos'],
                    'end_pos': current_segment['positions'][-1],
                    'avg_width_px': np.mean(current_segment['widths']),
                    'max_width_px': max(current_segment['widths']),
                    'length_frac': current_segment['positions'][-1] - current_segment['start_pos']
                })
        
        log.debug("Extracted %d segments from contour", len(segments))
        
    except Exception as e:
        log.warning("Segment extraction failed: %s", e)
    
    return segments


def _measure_perpendicular_width(mask: np.ndarray, x: int, y: int, angle: float, max_dist: int = 200) -> int:
    """
    Measure the width of the object perpendicular to the main axis at a given point.
    """
    # Perpendicular direction
    perp_angle = angle + 90
    rad = math.radians(perp_angle)
    dx = math.cos(rad)
    dy = math.sin(rad)
    
    h, w_img = mask.shape
    
    # Search in both perpendicular directions
    dist_pos = 0
    dist_neg = 0
    
    for dist in range(1, max_dist):
        px = int(x + dist * dx)
        py = int(y + dist * dy)
        if 0 <= px < w_img and 0 <= py < h:
            if mask[py, px] == 0:
                dist_pos = dist
                break
        else:
            dist_pos = dist
            break
    
    for dist in range(1, max_dist):
        px = int(x - dist * dx)
        py = int(y - dist * dy)
        if 0 <= px < w_img and 0 <= py < h:
            if mask[py, px] == 0:
                dist_neg = dist
                break
        else:
            dist_neg = dist
            break
    
    return dist_pos + dist_neg


def _analyze_segments(segments: list, px_per_mm: float) -> tuple:
    """
    Analyze segments to identify handle (wide part) and shaft (thin part).
    Returns: (handle_segment, shaft_segment) or (None, None)
    """
    if not segments:
        return None, None
    
    # Sort segments by width (widest first)
    sorted_segs = sorted(segments, key=lambda s: s.get('max_width_px', 0), reverse=True)
    
    handle_seg = None
    shaft_seg = None
    
    if len(sorted_segs) >= 1:
        # Widest segment is likely the handle
        handle = sorted_segs[0]
        handle_seg = {
            'width_mm': handle['max_width_px'] / px_per_mm if px_per_mm > 0 else 0,
            'height_mm': handle['avg_width_px'] / px_per_mm if px_per_mm > 0 else 0,
            'length_frac': handle.get('length_frac', 0)
        }
    
    if len(sorted_segs) >= 2:
        # Second widest (or thinner) is the shaft
        shaft = sorted_segs[-1]  # Take thinnest
        shaft_seg = {
            'width_mm': shaft['length_frac'] * 100,  # Approximate based on fraction
            'height_mm': shaft['avg_width_px'] / px_per_mm if px_per_mm > 0 else 0,
        }
    
    return handle_seg, shaft_seg


def _draw_measurements_on_image(
    img: np.ndarray, 
    cx: int, cy: int, 
    w_px: int, h_px: int, 
    angle: float,
    width_mm: float, 
    depth_mm: float,
    segments: list,
    px_per_mm: float
) -> None:
    """
    Draw measurement annotations on the debug image.
    """
    # Draw main dimension labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color_length = (0, 255, 0)  # Green for length
    color_width = (255, 165, 0)  # Orange for width
    color_segment = (0, 255, 255)  # Yellow for segments
    
    # Calculate label positions based on angle
    angle_rad = math.radians(angle)
    
    # Length label (along main axis)
    label_length = f"L={width_mm:.2f}cm"
    lx = int(cx + (w_px / 4) * math.cos(angle_rad))
    ly = int(cy + (w_px / 4) * math.sin(angle_rad) - 20)
    cv2.putText(img, label_length, (lx, ly), font, font_scale, color_length, thickness)
    
    # Draw arrow for length
    p1 = (int(cx - (w_px / 2) * math.cos(angle_rad)), 
          int(cy - (w_px / 2) * math.sin(angle_rad)))
    p2 = (int(cx + (w_px / 2) * math.cos(angle_rad)), 
          int(cy + (w_px / 2) * math.sin(angle_rad)))
    cv2.arrowedLine(img, p1, p2, color_length, 2, tipLength=0.05)
    cv2.arrowedLine(img, p2, p1, color_length, 2, tipLength=0.05)
    
    # Width label (perpendicular)
    label_width = f"W={depth_mm:.2f}cm"
    perp_angle_rad = angle_rad + math.pi / 2
    wx = int(cx + (h_px / 2 + 30) * math.cos(perp_angle_rad))
    wy = int(cy + (h_px / 2 + 30) * math.sin(perp_angle_rad))
    cv2.putText(img, label_width, (wx, wy), font, font_scale, color_width, thickness)
    
    # Draw width arrow
    p3 = (int(cx - (h_px / 2) * math.cos(perp_angle_rad)),
          int(cy - (h_px / 2) * math.sin(perp_angle_rad)))
    p4 = (int(cx + (h_px / 2) * math.cos(perp_angle_rad)),
          int(cy + (h_px / 2) * math.sin(perp_angle_rad)))
    cv2.arrowedLine(img, p3, p4, color_width, 2, tipLength=0.1)
    cv2.arrowedLine(img, p4, p3, color_width, 2, tipLength=0.1)
    
    # Draw segment information
    if segments:
        for i, seg in enumerate(segments):
            seg_width_mm = seg.get('max_width_px', 0) / px_per_mm if px_per_mm > 0 else 0
            # Position based on segment location
            seg_x = int(cx + seg.get('start_pos', 0) * w_px * math.cos(angle_rad))
            seg_y = int(cy + seg.get('start_pos', 0) * w_px * math.sin(angle_rad))
            label = f"S{i+1}={seg_width_mm:.2f}"
            cv2.putText(img, label, (seg_x, seg_y - 30), font, 0.6, color_segment, 1)
