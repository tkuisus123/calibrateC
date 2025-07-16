#!/usr/bin/env python3
"""
Image Mosaicking Script with Hough Transform-based Straightening

This script stitches video frames together to create a larger mosaic and applies
automatic straightening using Hough Transform to detect and correct rotational drift.

Usage:
    python image_mosaic.py [options]

Features:
    - Video frame extraction and mosaicking
    - Automatic rotational drift correction using Hough Transform
    - Edge detection with cv2.Canny
    - Line detection with cv2.HoughLinesP
    - Median angle calculation for robust tilt estimation
    - Image rotation with cv2.warpAffine preserving full content

Author: Auto-generated for calibrateC repository
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import math


def straighten_image_with_hough(image):
    """
    Straighten an image using Hough Transform to detect prominent vertical lines.
    
    This function detects vertical lines in the image and calculates the median
    angle to determine the overall tilt. It then applies a rotation transformation
    to make the lines perfectly vertical.
    
    Args:
        image (numpy.ndarray): Input mosaic image to be straightened
        
    Returns:
        tuple: (straightened_image, rotation_angle, line_info)
            - straightened_image: The corrected image with vertical lines
            - rotation_angle: The angle (in degrees) that was applied for correction
            - line_info: Dictionary with detection statistics
    """
    if image is None or image.size == 0:
        return image, 0, {"error": "Invalid input image"}
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    # Adaptive thresholds based on image statistics
    mean_intensity = np.mean(blurred)
    low_threshold = max(50, int(mean_intensity * 0.5))
    high_threshold = min(200, int(mean_intensity * 1.5))
    
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Line detection using HoughLinesP
    # Parameters tuned for detecting prominent lines
    rho = 1  # Distance resolution in pixels
    theta = np.pi / 180  # Angular resolution in radians (1 degree)
    threshold = max(50, min(image.shape[0], image.shape[1]) // 4)  # Minimum votes
    min_line_length = max(100, min(image.shape[0], image.shape[1]) // 8)  # Minimum line length
    max_line_gap = 20  # Maximum gap between line segments
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, 
                           minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    line_info = {
        "total_lines_detected": 0,
        "vertical_lines": 0,
        "angles": [],
        "median_angle": 0,
        "edge_pixels": np.sum(edges > 0),
        "canny_thresholds": (low_threshold, high_threshold)
    }
    
    if lines is None or len(lines) == 0:
        print("Warning: No lines detected in image")
        return image, 0, line_info
    
    line_info["total_lines_detected"] = len(lines)
    
    # Calculate angles of detected lines and filter for nearly vertical ones
    angles = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle of the line relative to horizontal
        if x2 - x1 == 0:
            # Perfectly vertical line
            angle = 90.0
        else:
            # Calculate angle in degrees
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Normalize angle to [-90, 90] range
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        
        # Filter for nearly vertical lines
        # Vertical lines should have angles close to ±90 degrees from horizontal
        # or close to 0 degrees when normalized to [-90, 90]
        abs_angle = abs(angle)
        if abs_angle > 60 or abs_angle < 30:  # Lines that are more vertical than horizontal
            # Convert to deviation from vertical (0 degrees = perfectly vertical)
            if abs_angle > 60:
                # Line is close to vertical (±90°), convert to deviation from vertical
                vertical_deviation = 90 - abs_angle if angle > 0 else -90 - angle
            else:
                # Line is close to horizontal but we treat small angles as nearly vertical
                vertical_deviation = angle
            
            angles.append(vertical_deviation)
            vertical_lines.append(line[0])
    
    line_info["vertical_lines"] = len(vertical_lines)
    line_info["angles"] = angles
    
    if len(angles) < 3:
        print(f"Warning: Only {len(angles)} nearly vertical lines found. Using original image.")
        return image, 0, line_info
    
    # Calculate median angle for robust estimation (less sensitive to outliers)
    median_angle = np.median(angles)
    line_info["median_angle"] = median_angle
    
    # Convert to rotation angle (angle to rotate to make lines vertical)
    # If lines are tilted, we need to rotate in the opposite direction
    rotation_angle = -median_angle  # Negative because we want to counter the tilt
    
    # Limit rotation to reasonable range
    if abs(rotation_angle) > 45:
        print(f"Warning: Large rotation angle detected ({rotation_angle:.2f}°). Limiting to ±45°.")
        rotation_angle = np.sign(rotation_angle) * min(abs(rotation_angle), 45)
    
    print(f"Detected tilt: {median_angle:.2f}°, applying correction: {rotation_angle:.2f}°")
    print(f"Analysis: {len(vertical_lines)} vertical lines from {len(lines)} total lines")
    
    # Apply rotation transformation using cv2.warpAffine
    height, width = image.shape[:2]
    
    # Calculate rotation matrix around image center
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Calculate new bounding box to ensure no content is cropped
    # Get corner points of the original image
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T
    
    # Transform corner points
    transformed_corners = rotation_matrix @ corners
    
    # Find the new bounding box
    x_coords = transformed_corners[0, :]
    y_coords = transformed_corners[1, :]
    
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    # Calculate new image dimensions
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))
    
    # Adjust translation to ensure all content is visible
    rotation_matrix[0, 2] += -min_x
    rotation_matrix[1, 2] += -min_y
    
    # Apply the rotation with the new dimensions
    straightened_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))
    
    print(f"Original size: {width}x{height}, Straightened size: {new_width}x{new_height}")
    
    return straightened_image, rotation_angle, line_info


def create_simple_mosaic(video_path, output_dir, max_frames=None):
    """
    Create a simple mosaic from video frames for demonstration purposes.
    
    This is a basic implementation that arranges frames in a grid pattern.
    In a real application, this would include sophisticated stitching algorithms.
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save output files
        max_frames (int): Maximum number of frames to process (None for all)
        
    Returns:
        tuple: (rgb_mosaic, depth_mosaic) - The created mosaics
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"Total frames available: {total_frames}")
    
    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
            
        if frame_count % 50 == 0:
            print(f"Extracted {frame_count} frames...")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
        
    print(f"Extracted {len(frames)} frames for mosaicking")
    
    # Create a simple grid mosaic
    # Calculate grid dimensions
    n_frames = len(frames)
    grid_cols = int(np.ceil(np.sqrt(n_frames)))
    grid_rows = int(np.ceil(n_frames / grid_cols))
    
    # Get frame dimensions
    frame_height, frame_width = frames[0].shape[:2]
    
    # Create mosaic canvas
    mosaic_width = grid_cols * frame_width
    mosaic_height = grid_rows * frame_height
    
    # Create RGB mosaic
    rgb_mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    
    # Create depth mosaic (simulated - convert to grayscale for demonstration)
    depth_mosaic = np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)
    
    # Fill the mosaic grid
    for i, frame in enumerate(frames):
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate position in mosaic
        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width
        
        # Place frame in mosaic
        rgb_mosaic[y_start:y_end, x_start:x_end] = frame
        
        # Create depth map (convert to grayscale and apply some processing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_mosaic[y_start:y_end, x_start:x_end] = gray
    
    print(f"Created mosaic: {mosaic_width}x{mosaic_height} pixels")
    print(f"Grid arrangement: {grid_rows}x{grid_cols}")
    
    return rgb_mosaic, depth_mosaic


def save_mosaic_results(rgb_mosaic, depth_mosaic, output_dir, timestamp):
    """
    Save mosaic results and analysis information.
    
    Args:
        rgb_mosaic (numpy.ndarray): RGB mosaic image
        depth_mosaic (numpy.ndarray): Depth mosaic image  
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
        
    Returns:
        dict: Dictionary of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Save original mosaics
    rgb_path = os.path.join(output_dir, f"rgb_mosaic_{timestamp}.png")
    depth_path = os.path.join(output_dir, f"depth_mosaic_{timestamp}.png")
    
    cv2.imwrite(rgb_path, rgb_mosaic)
    cv2.imwrite(depth_path, depth_mosaic)
    
    saved_files['original_rgb'] = rgb_path
    saved_files['original_depth'] = depth_path
    
    print(f"Saved original RGB mosaic: {rgb_path}")
    print(f"Saved original depth mosaic: {depth_path}")
    
    return saved_files


def main():
    """
    Main function demonstrating the image mosaicking with Hough Transform straightening.
    """
    parser = argparse.ArgumentParser(description='Image Mosaicking with Hough Transform Straightening')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default='mosaic_output', help='Output directory')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--demo', action='store_true', help='Run with demo data if no video specified')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMAGE MOSAICKING WITH HOUGH TRANSFORM STRAIGHTENING")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if args.video:
        # Process specified video
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
            
        print(f"Processing video: {video_path}")
        
        try:
            # Create mosaics from video
            final_rgb_mosaic, final_depth_mosaic = create_simple_mosaic(
                video_path, output_dir, args.max_frames
            )
            
        except Exception as e:
            print(f"Error creating mosaic: {e}")
            return
            
    elif args.demo:
        # Create demo mosaic for testing
        print("Creating demo mosaic for testing...")
        
        # Create a demo image with intentional tilt for testing
        demo_width, demo_height = 800, 600
        demo_image = np.ones((demo_height, demo_width, 3), dtype=np.uint8) * 255
        
        # Draw some vertical lines with a tilt to simulate drift
        tilt_angle = 5  # 5 degree tilt
        center_x, center_y = demo_width // 2, demo_height // 2
        
        # Draw several vertical lines at different positions
        for x_offset in range(-200, 201, 100):
            # Calculate line endpoints with tilt
            x_start = center_x + x_offset
            y_start = 50
            y_end = demo_height - 50
            
            # Apply tilt
            line_length = y_end - y_start
            x_end = x_start + int(line_length * math.tan(math.radians(tilt_angle)))
            
            cv2.line(demo_image, (x_start, y_start), (x_end, y_end), (0, 0, 0), 3)
        
        # Add some horizontal reference lines
        for y_offset in range(100, demo_height, 150):
            cv2.line(demo_image, (50, y_offset), (demo_width - 50, y_offset), (128, 128, 128), 2)
        
        # Add some noise and texture
        noise = np.random.randint(0, 50, (demo_height, demo_width, 3), dtype=np.uint8)
        demo_image = cv2.addWeighted(demo_image, 0.9, noise, 0.1, 0)
        
        final_rgb_mosaic = demo_image
        final_depth_mosaic = cv2.cvtColor(demo_image, cv2.COLOR_BGR2GRAY)
        
        print(f"Created demo mosaic with {tilt_angle}° tilt")
        
    else:
        print("Error: Please specify --video path or use --demo for testing")
        print("Example: python image_mosaic.py --video video.mp4")
        print("Example: python image_mosaic.py --demo")
        return
    
    # Save original mosaics
    print("\n" + "=" * 40)
    print("SAVING ORIGINAL MOSAICS")
    print("=" * 40)
    
    saved_files = save_mosaic_results(final_rgb_mosaic, final_depth_mosaic, output_dir, timestamp)
    
    # Apply Hough Transform straightening
    print("\n" + "=" * 40)
    print("APPLYING HOUGH TRANSFORM STRAIGHTENING")
    print("=" * 40)
    
    # Straighten RGB mosaic
    print("Straightening RGB mosaic...")
    straightened_rgb, rgb_angle, rgb_info = straighten_image_with_hough(final_rgb_mosaic)
    
    # Straighten depth mosaic
    print("Straightening depth mosaic...")
    straightened_depth, depth_angle, depth_info = straighten_image_with_hough(final_depth_mosaic)
    
    # Save straightened mosaics
    print("\n" + "=" * 40)
    print("SAVING STRAIGHTENED MOSAICS")
    print("=" * 40)
    
    straightened_rgb_path = os.path.join(output_dir, f"straightened_rgb_mosaic_{timestamp}.png")
    straightened_depth_path = os.path.join(output_dir, f"straightened_depth_mosaic_{timestamp}.png")
    
    cv2.imwrite(straightened_rgb_path, straightened_rgb)
    cv2.imwrite(straightened_depth_path, straightened_depth)
    
    saved_files['straightened_rgb'] = straightened_rgb_path
    saved_files['straightened_depth'] = straightened_depth_path
    
    print(f"Saved straightened RGB mosaic: {straightened_rgb_path}")
    print(f"Saved straightened depth mosaic: {straightened_depth_path}")
    
    # Save analysis report
    analysis_report = {
        'timestamp': timestamp,
        'rgb_analysis': rgb_info,
        'depth_analysis': depth_info,
        'rgb_rotation_angle': rgb_angle,
        'depth_rotation_angle': depth_angle,
        'saved_files': saved_files
    }
    
    report_path = os.path.join(output_dir, f"straightening_report_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write("IMAGE STRAIGHTENING ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("RGB MOSAIC ANALYSIS:\n")
        f.write(f"  Rotation applied: {rgb_angle:.2f}°\n")
        f.write(f"  Lines detected: {rgb_info.get('total_lines_detected', 0)}\n")
        f.write(f"  Vertical lines: {rgb_info.get('vertical_lines', 0)}\n")
        f.write(f"  Median angle: {rgb_info.get('median_angle', 0):.2f}°\n\n")
        
        f.write("DEPTH MOSAIC ANALYSIS:\n")
        f.write(f"  Rotation applied: {depth_angle:.2f}°\n")
        f.write(f"  Lines detected: {depth_info.get('total_lines_detected', 0)}\n")
        f.write(f"  Vertical lines: {depth_info.get('vertical_lines', 0)}\n")
        f.write(f"  Median angle: {depth_info.get('median_angle', 0):.2f}°\n\n")
        
        f.write("SAVED FILES:\n")
        for file_type, path in saved_files.items():
            f.write(f"  {file_type}: {path}\n")
    
    print(f"Saved analysis report: {report_path}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"RGB mosaic correction: {rgb_angle:.2f}° rotation applied")
    print(f"Depth mosaic correction: {depth_angle:.2f}° rotation applied")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    if args.demo:
        print("\nDemo completed successfully!")
        print("You can now test with real video files using:")
        print(f"python {__file__} --video your_video.mp4")


if __name__ == "__main__":
    main()