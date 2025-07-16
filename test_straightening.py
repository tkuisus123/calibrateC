#!/usr/bin/env python3
"""
Comprehensive test suite for the Hough Transform-based image straightening functionality.

This script tests various scenarios to validate the robustness and accuracy of the
straighten_image_with_hough() function implementation.

Usage:
    python test_straightening.py
"""

import cv2
import numpy as np
import os
import math
from datetime import datetime
import sys

# Import our straightening function
try:
    from main9 import straighten_image_with_hough
    print("Successfully imported straighten_image_with_hough from main9.py")
except ImportError:
    try:
        from image_mosaic import straighten_image_with_hough
        print("Successfully imported straighten_image_with_hough from image_mosaic.py")
    except ImportError:
        print("Error: Could not import straighten_image_with_hough function")
        sys.exit(1)


def create_test_image_with_lines(width=800, height=600, tilt_angle=0, num_lines=5):
    """
    Create a test image with vertical lines at a specified tilt angle.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels  
        tilt_angle: Angle to tilt the lines (degrees)
        num_lines: Number of vertical lines to draw
        
    Returns:
        numpy.ndarray: Test image with tilted lines
    """
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Calculate line positions
    line_positions = np.linspace(width * 0.2, width * 0.8, num_lines, dtype=int)
    
    # Draw vertical lines with tilt
    for x_pos in line_positions:
        # Calculate line endpoints with tilt
        y_start = int(height * 0.1)
        y_end = int(height * 0.9)
        line_length = y_end - y_start
        
        # Apply tilt
        x_offset = int(line_length * math.tan(math.radians(tilt_angle)))
        x_end = x_pos + x_offset
        
        # Draw the line
        cv2.line(image, (x_pos, y_start), (x_end, y_end), (0, 0, 0), 3)
    
    # Add some horizontal reference lines (should not affect vertical detection)
    for y_offset in range(int(height * 0.3), int(height * 0.8), int(height * 0.2)):
        cv2.line(image, (int(width * 0.1), y_offset), (int(width * 0.9), y_offset), (128, 128, 128), 2)
    
    # Add some noise and texture
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    image = cv2.addWeighted(image, 0.95, noise, 0.05, 0)
    
    return image


def test_angle_accuracy():
    """Test the accuracy of angle detection for various known tilts."""
    print("\n=== TESTING ANGLE DETECTION ACCURACY ===")
    
    test_angles = [0, 1, 2, 3, 5, 7, 10, 15, -2, -5, -10]
    results = []
    
    for test_angle in test_angles:
        print(f"\nTesting with {test_angle}° tilt...")
        
        # Create test image
        test_image = create_test_image_with_lines(tilt_angle=test_angle, num_lines=7)
        
        # Apply straightening
        straightened, detected_angle, info = straighten_image_with_hough(test_image)
        
        # Calculate error
        correction_error = abs(detected_angle + test_angle)  # How well did we correct the tilt
        
        results.append({
            'input_angle': test_angle,
            'detected_tilt': info.get('median_angle', 0),
            'correction_applied': detected_angle,
            'correction_error': correction_error,
            'lines_detected': info.get('total_lines_detected', 0),
            'vertical_lines': info.get('vertical_lines', 0)
        })
        
        print(f"  Input tilt: {test_angle:.1f}°")
        print(f"  Detected tilt: {info.get('median_angle', 0):.2f}°")
        print(f"  Correction applied: {detected_angle:.2f}°")
        print(f"  Correction error: {correction_error:.2f}°")
        print(f"  Lines used: {info.get('vertical_lines', 0)}/{info.get('total_lines_detected', 0)}")
    
    # Calculate statistics
    print(f"\n=== ACCURACY SUMMARY ===")
    errors = [r['correction_error'] for r in results]
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    accurate_tests = sum(1 for e in errors if e < 0.5)  # Within 0.5 degrees
    
    print(f"Average correction error: {avg_error:.2f}°")
    print(f"Maximum correction error: {max_error:.2f}°")
    print(f"Tests within 0.5°: {accurate_tests}/{len(results)} ({accurate_tests/len(results)*100:.1f}%)")
    
    return results


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== TESTING EDGE CASES ===")
    
    # Test 1: Empty image
    print("\nTest 1: Empty image")
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = straighten_image_with_hough(empty_image)
    print(f"  Empty image result: angle={result[1]:.2f}°, lines={result[2].get('total_lines_detected', 0)}")
    
    # Test 2: Image with no lines
    print("\nTest 2: Image with no lines (pure noise)")
    noise_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    result = straighten_image_with_hough(noise_image)
    print(f"  Noise image result: angle={result[1]:.2f}°, lines={result[2].get('total_lines_detected', 0)}")
    
    # Test 3: Image with only horizontal lines
    print("\nTest 3: Image with only horizontal lines")
    horizontal_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    for y in range(50, 151, 25):
        cv2.line(horizontal_image, (20, y), (280, y), (0, 0, 0), 2)
    result = straighten_image_with_hough(horizontal_image)
    print(f"  Horizontal lines result: angle={result[1]:.2f}°, vertical_lines={result[2].get('vertical_lines', 0)}")
    
    # Test 4: Very small image
    print("\nTest 4: Very small image")
    small_image = create_test_image_with_lines(width=50, height=50, tilt_angle=5)
    result = straighten_image_with_hough(small_image)
    print(f"  Small image result: angle={result[1]:.2f}°, lines={result[2].get('total_lines_detected', 0)}")
    
    # Test 5: Single line image
    print("\nTest 5: Single vertical line")
    single_line_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.line(single_line_image, (100, 20), (110, 180), (0, 0, 0), 3)  # Slightly tilted line
    result = straighten_image_with_hough(single_line_image)
    print(f"  Single line result: angle={result[1]:.2f}°, vertical_lines={result[2].get('vertical_lines', 0)}")


def test_content_preservation():
    """Test that image content is preserved during rotation."""
    print("\n=== TESTING CONTENT PRESERVATION ===")
    
    # Create test image with distinctive corners
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw corner markers
    cv2.circle(test_image, (50, 50), 20, (255, 0, 0), -1)  # Top-left: red
    cv2.circle(test_image, (550, 50), 20, (0, 255, 0), -1)  # Top-right: green
    cv2.circle(test_image, (50, 350), 20, (0, 0, 255), -1)  # Bottom-left: blue
    cv2.circle(test_image, (550, 350), 20, (255, 255, 0), -1)  # Bottom-right: yellow
    
    # Add some vertical lines with tilt
    for x in range(150, 451, 100):
        cv2.line(test_image, (x, 100), (x + 20, 300), (0, 0, 0), 3)
    
    # Apply straightening
    straightened, angle, info = straighten_image_with_hough(test_image)
    
    print(f"Original size: {test_image.shape[:2]}")
    print(f"Straightened size: {straightened.shape[:2]}")
    print(f"Rotation applied: {angle:.2f}°")
    
    # Check if corners are still visible (approximate check)
    # We can't check exact positions since the image was rotated, but we can check color presence
    colors_present = {
        'red': np.any(straightened[:,:,2] > 200),
        'green': np.any(straightened[:,:,1] > 200),
        'blue': np.any(straightened[:,:,0] > 200),
        'yellow': np.any((straightened[:,:,1] > 200) & (straightened[:,:,2] > 200))
    }
    
    print("Corner markers preserved:")
    for color, present in colors_present.items():
        print(f"  {color}: {'✓' if present else '✗'}")
    
    # Calculate area ratio to check for content loss
    original_area = test_image.shape[0] * test_image.shape[1]
    straightened_area = straightened.shape[0] * straightened.shape[1]
    area_ratio = straightened_area / original_area
    
    print(f"Area ratio (straightened/original): {area_ratio:.2f}")
    print(f"Content preservation: {'✓' if area_ratio >= 1.0 else '✗'}")


def test_different_image_sizes():
    """Test the function with different image sizes."""
    print("\n=== TESTING DIFFERENT IMAGE SIZES ===")
    
    sizes = [
        (100, 100),   # Small square
        (400, 200),   # Wide rectangle
        (200, 400),   # Tall rectangle
        (800, 600),   # Standard resolution
        (1200, 800),  # Large image
    ]
    
    for width, height in sizes:
        print(f"\nTesting {width}x{height} image...")
        
        test_image = create_test_image_with_lines(width, height, tilt_angle=5, num_lines=max(3, width//150))
        
        # Measure processing time
        start_time = datetime.now()
        straightened, angle, info = straighten_image_with_hough(test_image)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Lines detected: {info.get('total_lines_detected', 0)}")
        print(f"  Vertical lines: {info.get('vertical_lines', 0)}")
        print(f"  Correction: {angle:.2f}°")


def save_test_results(results, output_dir="test_output"):
    """Save test results and example images."""
    print(f"\n=== SAVING TEST RESULTS ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save accuracy test results
    with open(os.path.join(output_dir, "accuracy_test_results.txt"), 'w') as f:
        f.write("ANGLE DETECTION ACCURACY TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Input°\tDetected°\tCorrected°\tError°\tLines\tVertical\n")
        f.write("-" * 60 + "\n")
        
        for r in results:
            f.write(f"{r['input_angle']:>6.1f}\t{r['detected_tilt']:>8.2f}\t"
                   f"{r['correction_applied']:>9.2f}\t{r['correction_error']:>6.2f}\t"
                   f"{r['lines_detected']:>5d}\t{r['vertical_lines']:>8d}\n")
        
        # Calculate summary statistics
        errors = [r['correction_error'] for r in results]
        f.write(f"\nSUMMARY STATISTICS:\n")
        f.write(f"Average error: {np.mean(errors):.2f}°\n")
        f.write(f"Maximum error: {np.max(errors):.2f}°\n")
        f.write(f"Std deviation: {np.std(errors):.2f}°\n")
        f.write(f"Tests within 0.5°: {sum(1 for e in errors if e < 0.5)}/{len(errors)}\n")
    
    # Create example images
    print("Creating example images...")
    
    # Example 1: Before and after correction
    test_image = create_test_image_with_lines(tilt_angle=8, num_lines=6)
    straightened, angle, info = straighten_image_with_hough(test_image)
    
    cv2.imwrite(os.path.join(output_dir, "example_tilted.png"), test_image)
    cv2.imwrite(os.path.join(output_dir, "example_straightened.png"), straightened)
    
    # Example 2: Analysis visualization
    analysis_image = test_image.copy()
    cv2.putText(analysis_image, f"Original Tilt: 8.0°", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(analysis_image, f"Detected: {info.get('median_angle', 0):.1f}°", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(analysis_image, f"Correction: {angle:.1f}°", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    straightened_annotated = straightened.copy()
    cv2.putText(straightened_annotated, "Straightened", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(output_dir, "analysis_before.png"), analysis_image)
    cv2.imwrite(os.path.join(output_dir, "analysis_after.png"), straightened_annotated)
    
    print(f"Test results saved to {output_dir}/")


def main():
    """Run all tests."""
    print("HOUGH TRANSFORM STRAIGHTENING - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Run all tests
    accuracy_results = test_angle_accuracy()
    test_edge_cases()
    test_content_preservation()
    test_different_image_sizes()
    
    # Save results
    save_test_results(accuracy_results)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("Check test_output/ directory for detailed results and example images")


if __name__ == "__main__":
    main()