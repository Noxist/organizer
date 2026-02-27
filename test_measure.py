#!/usr/bin/env python3
"""
Test script for object measurement.

Usage:
    python test_measure.py <image_file> [grid_square_mm]

Example:
    python test_measure.py screwdriver.jpg 20

Expected values for the test screwdriver (from annotated image):
    - Dimension 1 = 2.00 cm (handle height/diameter)
    - Dimension 2 = 11.25 cm (shaft length)
    - Dimension 3 = 1.85 cm (handle width)
    - Total length ≈ 13.1 cm
"""

import sys
import os
import logging

# Setup logging - INFO level by default, DEBUG if -v flag
log_level = logging.DEBUG if '-v' in sys.argv else logging.INFO
logging.basicConfig(
    level=log_level,
    format="[%(name)s] %(levelname)s: %(message)s",
)

# Remove -v from argv if present
sys.argv = [a for a in sys.argv if a != '-v']

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure
import cv2


def test_measure(image_path: str, grid_square_mm: int = 20):
    """Test measurement on an image file."""
    
    print(f"\n{'='*60}")
    print(f"Testing measurement on: {image_path}")
    print(f"Grid square size: {grid_square_mm}mm ({grid_square_mm/10}cm)")
    print(f"{'='*60}\n")
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Also load image directly for debug analysis
    import numpy as np
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    # Analyze gray values in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("IMAGE ANALYSIS:")
    print("-" * 40)
    print(f"Image shape: {img.shape}")
    print(f"Gray value range: {gray.min()} - {gray.max()}")
    
    # Find non-white pixels (potential object)
    non_white = gray < 230
    print(f"Non-white pixels: {np.sum(non_white)} ({100*np.sum(non_white)/gray.size:.1f}%)")
    
    # Find gray range pixels (potential shaft)
    gray_pixels = (gray > 100) & (gray < 200) & (gray < 230)
    print(f"Gray range (100-200): {np.sum(gray_pixels)} pixels")
    print()
    
    try:
        result = measure.measure_object(image_bytes, grid_square_mm)
        
        print("MEASUREMENT RESULTS:")
        print("-" * 40)
        print(f"Width (length):  {result.width_mm:.1f} mm = {result.width_cm:.2f} cm")
        print(f"Depth (height):  {result.depth_mm:.1f} mm = {result.depth_cm:.2f} cm")
        print(f"Rotation angle:  {result.angle:.1f}°")
        print(f"Pixels per mm:   {result.px_per_mm:.2f}")
        print()
        print("EXTENDED MEASUREMENTS:")
        print("-" * 40)
        print(f"Length (main axis):     {result.length_mm:.2f} mm = {result.length_cm:.2f} cm")
        print(f"Handle width (approx):  {result.handle_width_mm:.2f} mm")
        print(f"Shaft length (approx):  {result.shaft_length_mm:.2f} mm")
        print()
        print("EXPECTED VALUES (from annotated image):")
        print("-" * 40)
        print("  1 = 2.00 cm (handle height)")
        print("  2 = 11.25 cm (shaft length)")
        print("  3 = 1.85 cm (handle width)")
        print("  Total length ≈ 13.1 cm")
        print()
        
        # Save debug image
        if result.debug_image is not None:
            debug_path = image_path.rsplit('.', 1)[0] + '_debug.jpg'
            cv2.imwrite(debug_path, result.debug_image)
            print(f"Debug image saved to: {debug_path}")
        
        return result
        
    except measure.MeasurementError as e:
        print(f"ERROR: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo image file specified!")
        print("Usage: python test_measure.py <image_file> [grid_square_mm]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    grid_square_mm = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    test_measure(image_path, grid_square_mm)


if __name__ == "__main__":
    main()
