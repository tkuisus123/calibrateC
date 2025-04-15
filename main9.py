import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import os
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class ExclusiveCameraTracker:
    def __init__(self, front_video_path, side_video_path, floor_video_path, config=None):
        """
        Initialize tracker with exclusive camera assignments:
        - Floor camera for X and Z coordinates
        - Front and side cameras for Y coordinate

        Args:
         front_video_path = "front7.mp4"  # Path to front camera video
         side_video_path = "side7.mp4"    # Path to side camera video
         floor_video_path = "flor7.mp4"   # Path to floor camera video
            config: Optional dictionary with configuration parameters
        """

        # Default configuration with exclusive camera assignments
        self.config = {
            # Color detection parameters
            'red_lower1': [0, 120, 70],  # Lower HSV threshold for red (first range)
            'red_upper1': [10, 255, 255],  # Upper HSV threshold for red (first range)
            'red_lower2': [170, 120, 70],  # Lower HSV threshold for red (second range)
            'red_upper2': [180, 255, 255],  # Upper HSV threshold for red (second range)
            'min_contour_area': 50,  # Minimum contour area to consider

            # Dimension weighting parameters - EXCLUSIVE ASSIGNMENTS
            'dimension_weights': {
                'X': {'front': 0.0, 'floor': 1.0, 'side': 0.0},  # Only floor camera for X
                'Y': {'front': 0.5, 'floor': 0.0, 'side': 0.5},  # Only front and side for Y
                'Z': {'front': 0.0, 'floor': 1.0, 'side': 0.0}  # Only floor camera for Z
            },

            # Camera alignment correction
            'camera_flip': {  # Flip axes if needed
                'front_x': False,
                'front_y': False,
                'side_x': False,
                'side_y': False,
                'floor_x': False,
                'floor_y': False
            },

            # Y-coordinate verification and calibration parameters
            'y_calibration_mode': 'automatic',  # 'automatic', 'manual', or 'disabled'
            'y_validation_window': 10,  # Number of frames to use for validation
            'y_disagreement_threshold': 0.2,  # Threshold for detecting disagreement (0-1)
            'y_movement_min_threshold': 0.05,  # Minimum movement to consider for validation
            'y_calibration_frames': 30,  # Frames to collect for initial calibration
            'enable_startup_calibration': False,  # Run a calibration phase at startup
            'y_correlation_method': 'pearson',  # 'pearson', 'spearman', or 'kendall'
            'show_overlay_comparison': True,  # Show overlay of camera data in visualization
            'camera_debug_info': True,  # Show debugging info on camera frames

            # Cross-camera analysis parameters
            'use_cross_camera_analysis': True,  # Use 3-camera cross-analysis
            'analysis_frames': 300,  # Number of frames to analyze
            'analysis_interval': 2,  # Sample every 2nd frame

            # Y-coordinate usage parameters
            'y_blending_method': 'adaptive',  # 'weighted', 'adaptive', 'best_confidence', 'average'
            'y_conflict_resolution': 'voting',  # 'voting', 'highest_confidence', 'most_recent'
            'highlight_y_conflicts': True,  # Highlight frames where Y coordinates conflict

            # Display and filtering parameters
            'smoothing_factor': 0.6,  # Smoothing factor (0-1)
            'outlier_threshold': 10.0,  # Distance in units to consider a point an outlier
            'display_scale': 100,  # Scaling factor for 3D coordinates
            'display_fps': 30,  # Target FPS for display
            'camera_width': 640,  # Display width for camera views
            'camera_height': 480,  # Display height for camera views
            'show_comparison_window': True  # Show separate window with camera comparison
        }

        # Update with user configuration if provided
        if config:
            # Deep update for nested dictionaries
            for key, value in config.items():
                if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value

        # Open video captures
        self.cap_front = cv2.VideoCapture(front_video_path)
        self.cap_side = cv2.VideoCapture(side_video_path)
        self.cap_floor = cv2.VideoCapture(floor_video_path)

        # Ensure video captures opened correctly
        captures = {
            'front': self.cap_front.isOpened(),
            'side': self.cap_side.isOpened(),
            'floor': self.cap_floor.isOpened()
        }
        if not all(captures.values()):
            failed = [k for k, v in captures.items() if not v]
            raise IOError(f"Could not open video file(s): {', '.join(failed)}")

        # Get video dimensions
        self.width_front = int(self.cap_front.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height_front = int(self.cap_front.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width_side = int(self.cap_side.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height_side = int(self.cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width_floor = int(self.cap_floor.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height_floor = int(self.cap_floor.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get FPS for timing calculations
        self.fps = self.cap_front.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0  # Default if FPS not available

        # Store tracking data
        self.trajectory_3d = []  # Final 3D trajectory
        self.raw_camera_data = []  # Raw 2D points from each camera
        self.camera_confidences = []  # Confidence scores for each camera
        self.timestamps = []  # Timestamp for each frame
        self.frame_count = 0  # Total frames processed

        # For smoothing and outlier detection
        self.prev_positions = []
        self.dimension_limits = {'X': [float('inf'), float('-inf')],
                                 'Y': [float('inf'), float('-inf')],
                                 'Z': [float('inf'), float('-inf')]}

        # For detecting primary plane of movement
        self.dimension_movements = {'X': [], 'Y': [], 'Z': []}

        # For Y-coordinate validation and calibration
        self.front_y_values = []  # History of front camera Y values
        self.side_y_values = []  # History of side camera Y values
        self.front_y_raw = []  # Raw Y values from front camera before direction correction
        self.side_y_raw = []  # Raw Y values from side camera before direction correction
        self.front_y_movements = []  # Frame-to-frame movements in front camera
        self.side_y_movements = []  # Frame-to-frame movements in side camera
        self.y_agreement_scores = []  # History of agreement between cameras
        self.y_correlation_scores = []  # Correlation coefficients between cameras
        self.y_conflict_frames = []  # Frames where Y coordinates conflict significantly

        # Camera direction states
        self.front_y_direction = 1  # Default direction (1 or -1)
        self.side_y_direction = 1  # Default direction (1 or -1)
        self.y_verification_state = "Waiting to start analysis"  # Current verification state

        # Calibration state
        self.calibration_active = self.config['enable_startup_calibration']
        self.calibration_frame_count = 0
        self.calibration_complete = False
        self.calibration_results = {}

        # Start timing
        self.start_time = time.time()

        # Setup visualization
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(16, 10))
        self.setup_visualization()

        # For text annotations in matplotlib
        self.text_annotations = []

        # Create dedicated comparison window if enabled
        self.comparison_window = None
        if self.config['show_comparison_window']:
            self.comparison_window = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.namedWindow('Y-Coordinate Comparison', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Y-Coordinate Comparison', 800, 600)

        # Print configuration
        print("Exclusive Camera Tracker initialized")
        print(f"Front camera: {self.width_front}x{self.height_front} - Controls Y coordinate")
        print(f"Side camera: {self.width_side}x{self.height_side} - Controls Y coordinate")
        print(f"Floor camera: {self.width_floor}x{self.height_floor} - Controls X and Z coordinates")
        print("Camera assignment mode: EXCLUSIVE with Y-coordinate verification")

        if self.config['use_cross_camera_analysis']:
            print("Y-direction analysis: CROSS-CAMERA ANALYSIS enabled")

    def setup_visualization(self):
        """Setup the visualization plots"""
        # Main 3D trajectory plot
        self.ax_3d = self.fig.add_subplot(231, projection='3d')
        self.ax_3d.set_xlabel('X (Floor)', fontsize=9)
        self.ax_3d.set_ylabel('Y (Front/Side)', fontsize=9)
        self.ax_3d.set_zlabel('Z (Floor)', fontsize=9)
        self.ax_3d.set_title('3D Trajectory', fontweight='bold')
        self.ax_3d.grid(True)

        # Top View - XZ Plane (Floor Camera)
        self.ax_top = self.fig.add_subplot(232)
        self.ax_top.set_xlabel('X')
        self.ax_top.set_ylabel('Z')
        self.ax_top.set_title('Top View (X-Z) - Floor Camera')
        self.ax_top.grid(True)

        # Front View - XY Plane (Front Camera)
        self.ax_front = self.fig.add_subplot(233)
        self.ax_front.set_xlabel('X')
        self.ax_front.set_ylabel('Y')
        self.ax_front.set_title('Front View (X-Y) - Front Camera')
        self.ax_front.grid(True)

        # Side View - ZY Plane (Side Camera)
        self.ax_side = self.fig.add_subplot(234)
        self.ax_side.set_xlabel('Z')
        self.ax_side.set_ylabel('Y')
        self.ax_side.set_title('Side View (Z-Y) - Side Camera')
        self.ax_side.grid(True)

        # Dimension Movement Analysis
        self.ax_movement = self.fig.add_subplot(235)
        self.ax_movement.set_xlabel('Frame')
        self.ax_movement.set_ylabel('Movement')
        self.ax_movement.set_title('Dimension Movement Analysis')
        self.ax_movement.grid(True)

        # Y-Coordinate Verification Plot with enhanced styling
        self.ax_y_verify = self.fig.add_subplot(236)
        self.ax_y_verify.set_xlabel('Frame')
        self.ax_y_verify.set_ylabel('Y Value')
        self.ax_y_verify.set_title('Y-Coordinate Verification', fontweight='bold')
        self.ax_y_verify.grid(True)

        # Add special instructions for Y-verification
        if self.config['use_cross_camera_analysis']:
            calibration_text = "Cross-camera analysis will run at startup"
        elif self.config['enable_startup_calibration']:
            calibration_text = "Camera calibration will run at startup"
        else:
            calibration_text = "Y-coordinate verification is active"

        self.ax_y_verify.text(0.5, 0.5, calibration_text,
                              ha='center', va='center', transform=self.ax_y_verify.transAxes,
                              bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                              fontsize=10)

        # Initialize plot elements with better styling
        self.line_3d, = self.ax_3d.plot3D([], [], [], 'r-', linewidth=2)
        self.point_3d = self.ax_3d.scatter([], [], [], color='blue', s=50)

        self.line_top, = self.ax_top.plot([], [], 'b-')
        self.point_top = self.ax_top.scatter([], [], color='red', s=30)

        self.line_front, = self.ax_front.plot([], [], 'g-')
        self.point_front = self.ax_front.scatter([], [], color='red', s=30)

        self.line_side, = self.ax_side.plot([], [], 'm-')
        self.point_side = self.ax_side.scatter([], [], color='red', s=30)

        # Movement analysis lines
        self.line_x_movement, = self.ax_movement.plot([], [], 'r-', label='X')
        self.line_y_movement, = self.ax_movement.plot([], [], 'g-', label='Y')
        self.line_z_movement, = self.ax_movement.plot([], [], 'b-', label='Z')
        self.ax_movement.legend()

        # Y-coordinate verification lines with enhanced styling
        self.line_front_y, = self.ax_y_verify.plot([], [], 'g-', label='Front Y', linewidth=2)
        self.line_side_y, = self.ax_y_verify.plot([], [], 'm-', label='Side Y', linewidth=2)
        self.line_y_agreement, = self.ax_y_verify.plot([], [], 'k--', label='Agreement', alpha=0.7)

        # Add legend with better positioning
        self.ax_y_verify.legend(loc='upper right', fontsize=8)

        # Set graph background color to highlight Y-verification
        if self.config['highlight_y_conflicts']:
            self.ax_y_verify.set_facecolor('#f0f0f0')  # Light gray background

        # Add camera direction indicators for Y-coordinate
        front_arrow = plt.Arrow(0.05, 0.90, 0, -0.1, width=0.05, color='green',
                                transform=self.ax_y_verify.transAxes)
        self.ax_y_verify.add_patch(front_arrow)
        self.ax_y_verify.text(0.12, 0.87, "Front Y", transform=self.ax_y_verify.transAxes,
                              color='green', fontsize=8)

        side_arrow = plt.Arrow(0.05, 0.80, 0, -0.1, width=0.05, color='magenta',
                               transform=self.ax_y_verify.transAxes)
        self.ax_y_verify.add_patch(side_arrow)
        self.ax_y_verify.text(0.12, 0.77, "Side Y", transform=self.ax_y_verify.transAxes,
                              color='magenta', fontsize=8)

        # Add display method information
        if self.config['use_cross_camera_analysis']:
            mode_text = "Analysis: Cross-Camera"
        else:
            calibration_mode = self.config['y_calibration_mode']
            mode_text = f"Calibration: {calibration_mode}"

        self.ax_y_verify.text(0.5, 0.05, mode_text, ha='center', transform=self.ax_y_verify.transAxes,
                              fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()

    def detect_red_object(self, frame, source):
        """
        Detect a red object in the frame and calculate confidence

        Args:
            frame: Input camera frame
            source: Camera source ('front', 'side', or 'floor')

        Returns:
            tuple: (point, confidence, annotated_frame)
        """
        if frame is None:
            return None, 0.0, frame

        # Create a copy for drawing
        result_frame = frame.copy()

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get HSV thresholds from config
        lower_red1 = np.array(self.config['red_lower1'])
        upper_red1 = np.array(self.config['red_upper1'])
        lower_red2 = np.array(self.config['red_lower2'])
        upper_red2 = np.array(self.config['red_upper2'])

        # Create masks for both red ranges and combine
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of the red objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Skip very small contours that might be noise
            if area < self.config['min_contour_area']:
                cv2.putText(result_frame, f"No object ({source})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return None, 0.0, result_frame

            # Calculate centroid using moments
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Apply coordinate system adjustments from config
                if source == 'front':
                    if self.config['camera_flip']['front_x']:
                        cx = self.width_front - cx
                    if self.config['camera_flip']['front_y']:
                        cy = self.height_front - cy
                elif source == 'side':
                    if self.config['camera_flip']['side_x']:
                        cx = self.width_side - cx
                    if self.config['camera_flip']['side_y']:
                        cy = self.height_side - cy
                elif source == 'floor':
                    if self.config['camera_flip']['floor_x']:
                        cx = self.width_floor - cx
                    if self.config['camera_flip']['floor_y']:
                        cy = self.height_floor - cy

                # Calculate confidence based on object properties
                # 1. Size confidence - larger objects are more reliable
                frame_area = frame.shape[0] * frame.shape[1]
                size_conf = min(area / (frame_area * 0.05), 1.0)  # Expect object to be ~5% of frame

                # 2. Position confidence - objects in center are more reliable
                center_x = frame.shape[1] / 2
                center_y = frame.shape[0] / 2
                dist_from_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                position_conf = 1.0 - (dist_from_center / max_dist)

                # 3. Color confidence - how well it matches red
                color_conf = 0.8  # Default high confidence for simplicity

                # Combine confidences with weights
                confidence = 0.5 * size_conf + 0.3 * position_conf + 0.2 * color_conf

                # Draw the contour and centroid
                cv2.drawContours(result_frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(result_frame, (cx, cy), 7, (0, 0, 255), -1)

                # Add information to the frame
                cv2.putText(result_frame, f"({cx}, {cy})",
                            (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

                # Add confidence
                cv2.putText(result_frame, f"Conf: {confidence:.2f}",
                            (cx - 30, cy + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

                # Add camera label and controlled coordinates
                if source == 'front':
                    control_txt = "Controls: Y" + (f" (Dir: {self.front_y_direction})" if self.frame_count > 10 else "")
                elif source == 'side':
                    control_txt = "Controls: Y" + (f" (Dir: {self.side_y_direction})" if self.frame_count > 10 else "")
                else:  # floor
                    control_txt = "Controls: X, Z"

                cv2.putText(result_frame, f"{source.upper()} - {control_txt}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                return (cx, cy), confidence, result_frame

        # No object detected
        cv2.putText(result_frame, f"No object ({source})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, 0.0, result_frame

    def analyze_three_camera_movement(self, frames_to_analyze=300, sample_interval=2):
        """
        Enhanced analysis that cross-references all three cameras to identify and validate
        Y-axis movement. This method leverages the relationship between all cameras to ensure
        proper Y-direction determination.

        Args:
            frames_to_analyze: Maximum number of frames to analyze
            sample_interval: Sample every Nth frame to speed up analysis

        Returns:
            bool: True if analysis was successful, False otherwise
        """
        try:
            print("\n[CROSS-CAM ANALYSIS] Analyzing movement across all three cameras...")
            print(f"[CROSS-CAM ANALYSIS] Will analyze up to {frames_to_analyze} frames...")

            # Initialize data collection arrays for all cameras
            front_data = []  # Each element is (frame_num, x, y, confidence)
            side_data = []  # Each element is (frame_num, x, y, confidence)
            floor_data = []  # Each element is (frame_num, x, y, confidence)

            # Track frame number for sampling
            frame_num = 0
            analyzed_frames = 0

            # Save original video positions
            front_pos = self.cap_front.get(cv2.CAP_PROP_POS_FRAMES)
            side_pos = self.cap_side.get(cv2.CAP_PROP_POS_FRAMES)
            floor_pos = self.cap_floor.get(cv2.CAP_PROP_POS_FRAMES)

            # Reset video positions to start
            self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap_floor.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Process frames to collect movement data
            progress_interval = frames_to_analyze // 10
            print("[CROSS-CAM ANALYSIS] Scanning video frames across all cameras...")

            while analyzed_frames < frames_to_analyze:
                # Read frames
                ret_front, frame_front = self.cap_front.read()
                ret_side, frame_side = self.cap_side.read()
                ret_floor, frame_floor = self.cap_floor.read()

                if not all([ret_front, ret_side, ret_floor]):
                    print(f"[CROSS-CAM ANALYSIS] Reached end of videos after {frame_num} frames")
                    break

                frame_num += 1

                # Only analyze every Nth frame to speed up processing
                if frame_num % sample_interval != 0:
                    continue

                analyzed_frames += 1
                if analyzed_frames % progress_interval == 0:
                    progress = int(analyzed_frames / frames_to_analyze * 100)
                    print(f"[CROSS-CAM ANALYSIS] Progress: {progress}% ({analyzed_frames}/{frames_to_analyze})")

                # Detect object in each frame
                try:
                    front_result = self.detect_red_object(frame_front, 'front')
                    side_result = self.detect_red_object(frame_side, 'side')
                    floor_result = self.detect_red_object(frame_floor, 'floor')

                    front_point, front_conf, _ = front_result
                    side_point, side_conf, _ = side_result
                    floor_point, floor_conf, _ = floor_result

                    # Store all detection data with normalization
                    if front_point is not None:
                        # Normalize coordinates to [0,1] range
                        x_norm = front_point[0] / self.width_front
                        y_norm = front_point[1] / self.height_front
                        front_data.append((frame_num, x_norm, y_norm, front_conf))

                    if side_point is not None:
                        # Normalize coordinates to [0,1] range
                        x_norm = side_point[0] / self.width_side
                        y_norm = side_point[1] / self.height_side
                        side_data.append((frame_num, x_norm, y_norm, side_conf))

                    if floor_point is not None:
                        # Normalize coordinates to [0,1] range
                        x_norm = floor_point[0] / self.width_floor
                        y_norm = floor_point[1] / self.height_floor
                        floor_data.append((frame_num, x_norm, y_norm, floor_conf))

                except Exception as e:
                    print(f"[CROSS-CAM ANALYSIS] Error processing frame {frame_num}: {e}")
                    continue

            # Restore original video positions
            self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, front_pos)
            self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, side_pos)
            self.cap_floor.set(cv2.CAP_PROP_POS_FRAMES, floor_pos)

            # Check if we have enough data points
            if len(front_data) < 10 or len(side_data) < 10 or len(floor_data) < 10:
                print(f"[CROSS-CAM ANALYSIS] Insufficient data points:")
                print(f"  Front camera: {len(front_data)} points")
                print(f"  Side camera: {len(side_data)} points")
                print(f"  Floor camera: {len(floor_data)} points")
                print("[CROSS-CAM ANALYSIS] Using default directions (front=1, side=1)")
                return False

            print(f"[CROSS-CAM ANALYSIS] Collected data from all cameras:")
            print(f"  Front camera: {len(front_data)} points")
            print(f"  Side camera: {len(side_data)} points")
            print(f"  Floor camera: {len(floor_data)} points")

            # Find segments with simultaneous detection in all cameras
            triple_segments = self.find_triple_segments(front_data, side_data, floor_data)

            if not triple_segments or len(triple_segments) < 2:
                print("[CROSS-CAM ANALYSIS] Insufficient segments with all cameras detecting the object")

                # Fall back to two-camera analysis between front and side
                print("[CROSS-CAM ANALYSIS] Falling back to front-side camera analysis...")
                common_segments = self.find_common_segments(
                    [(f, y) for f, _, y, _ in front_data],
                    [(f, x) for f, x, _, _ in side_data]  # x in side view is Y in 3D
                )

                if not common_segments or len(common_segments) < 2:
                    print("[CROSS-CAM ANALYSIS] Insufficient common segments between front and side cameras")
                    print("[CROSS-CAM ANALYSIS] Using default directions (front=1, side=1)")
                    return False

                return self.analyze_front_side_segments(common_segments)

            print(f"[CROSS-CAM ANALYSIS] Found {len(triple_segments)} segments with all three cameras")

            # Analyze Y-direction using all three cameras
            return self.analyze_triple_segments(triple_segments)

        except Exception as e:
            print(f"[CROSS-CAM ANALYSIS] Error during movement analysis: {e}")
            import traceback
            traceback.print_exc()

            # Reset to default in case of error
            self.front_y_direction = 1
            self.side_y_direction = 1
            self.calibration_active = False
            self.calibration_complete = True
            self.y_verification_state = "Cross-camera analysis failed, using defaults"

            return False

    def find_triple_segments(self, front_data, side_data, floor_data, min_length=5):
        """
        Find segments where all three cameras detected the object simultaneously.

        Args:
            front_data: List of (frame_num, x, y, conf) for front camera
            side_data: List of (frame_num, x, y, conf) for side camera
            floor_data: List of (frame_num, x, y, conf) for floor camera
            min_length: Minimum number of frames per segment

        Returns:
            list: List of (front_segment, side_segment, floor_segment) tuples
        """
        # Convert to frame-indexed dictionaries for faster lookup
        front_dict = {frame: (x, y, conf) for frame, x, y, conf in front_data}
        side_dict = {frame: (x, y, conf) for frame, x, y, conf in side_data}
        floor_dict = {frame: (x, y, conf) for frame, x, y, conf in floor_data}

        # Find frames where all three cameras detected the object
        common_frames = sorted(set(front_dict.keys()) & set(side_dict.keys()) & set(floor_dict.keys()))

        if not common_frames:
            return []

        # Split into continuous segments
        segments = []
        current_segment = []
        prev_frame = None

        for frame in common_frames:
            if prev_frame is None or frame == prev_frame + 1:
                # Continuous frame
                current_segment.append(frame)
            else:
                # Gap detected, start a new segment if current one is long enough
                if len(current_segment) >= min_length:
                    segments.append(current_segment)
                current_segment = [frame]
            prev_frame = frame

        # Add the last segment if it's long enough
        if len(current_segment) >= min_length:
            segments.append(current_segment)

        # Convert segments to triplets of camera data
        triple_segments = []
        for segment in segments:
            front_segment = [(frame, *front_dict[frame]) for frame in segment]
            side_segment = [(frame, *side_dict[frame]) for frame in segment]
            floor_segment = [(frame, *floor_dict[frame]) for frame in segment]
            triple_segments.append((front_segment, side_segment, floor_segment))

        return triple_segments

    def analyze_triple_segments(self, triple_segments):
        """
        Analyze Y-direction correlation using data from all three cameras.
        This leverages the relationship between all cameras to validate Y-direction.

        Args:
            triple_segments: List of (front_segment, side_segment, floor_segment) tuples

        Returns:
            bool: True if analysis successful, False otherwise
        """
        print("[CROSS-CAM ANALYSIS] Analyzing Y-direction using all three cameras...")

        # Results storage
        results = []

        for segment_idx, (front_segment, side_segment, floor_segment) in enumerate(triple_segments):
            print(f"[CROSS-CAM ANALYSIS] Analyzing segment {segment_idx + 1} ({len(front_segment)} frames)")

            # Extract frame-to-frame movements in front camera (Y is the 3rd value)
            front_y_vals = [y for _, _, y, _ in front_segment]
            front_y_diffs = [y2 - y1 for y1, y2 in zip(front_y_vals[:-1], front_y_vals[1:])]

            # Extract frame-to-frame movements in side camera (X is Y in 3D, 2nd value)
            side_x_vals = [x for _, x, _, _ in side_segment]
            side_x_diffs = [x2 - x1 for x1, x2 in zip(side_x_vals[:-1], side_x_vals[1:])]

            # Extract movements in floor camera - mainly for validation
            # In floor camera, X and Y correspond to X and Z in 3D
            floor_x_vals = [x for _, x, _, _ in floor_segment]
            floor_y_vals = [y for _, _, y, _ in floor_segment]
            floor_x_diffs = [x2 - x1 for x1, x2 in zip(floor_x_vals[:-1], floor_x_vals[1:])]
            floor_y_diffs = [y2 - y1 for y1, y2 in zip(floor_y_vals[:-1], floor_y_vals[1:])]

            # Filter for significant movements in Y axis (front camera) and X axis (side camera)
            threshold = self.config.get('y_movement_min_threshold', 0.01)

            # Identify significant Y-movements in the front camera
            significant_y_movements = [(i, front_y_diffs[i]) for i in range(len(front_y_diffs))
                                       if abs(front_y_diffs[i]) > threshold]

            if len(significant_y_movements) < 3:
                print(f"[CROSS-CAM ANALYSIS] Segment {segment_idx + 1}: Not enough significant Y movements")
                continue

            # For each significant Y movement, check corresponding X movement in side camera
            y_direction_pairs = []
            for idx, front_y_diff in significant_y_movements:
                if idx < len(side_x_diffs):
                    side_x_diff = side_x_diffs[idx]

                    # Only compare if side camera also shows significant movement
                    if abs(side_x_diff) > threshold:
                        y_direction_pairs.append((front_y_diff, side_x_diff))

            if len(y_direction_pairs) < 3:
                print(f"[CROSS-CAM ANALYSIS] Segment {segment_idx + 1}: Not enough paired movements")
                continue

            # Calculate agreement ratio (how often front Y and side X move in same/opposite directions)
            agreement_count = sum(1 for fy, sx in y_direction_pairs if (fy * sx) > 0)
            disagreement_count = sum(1 for fy, sx in y_direction_pairs if (fy * sx) < 0)

            total_count = agreement_count + disagreement_count
            if total_count > 0:
                agreement_ratio = agreement_count / total_count
            else:
                agreement_ratio = 0.5  # Neutral if no clear movements

            # Calculate correlation coefficient
            try:
                front_moves = np.array([fy for fy, _ in y_direction_pairs])
                side_moves = np.array([sx for _, sx in y_direction_pairs])
                correlation = np.corrcoef(front_moves, side_moves)[0, 1]
            except:
                correlation = 0

            # Cross-validation with floor camera
            # In a well-calibrated system, when object moves primarily in Y:
            # 1. We should see strong movement in front camera Y and side camera X
            # 2. Floor camera should show minimal movement (as it detects X-Z plane)

            # Calculate average movement magnitudes for validation
            y_move_magnitude = np.mean([abs(diff) for diff in front_y_diffs])
            side_x_move_magnitude = np.mean([abs(diff) for diff in side_x_diffs])
            floor_move_magnitude = np.mean(
                [abs(diff) for diff in floor_x_diffs] + [abs(diff) for diff in floor_y_diffs])

            # Y-dominance ratio (higher means movement is primarily in Y-axis)
            y_dominance = (y_move_magnitude + side_x_move_magnitude) / (floor_move_magnitude + 0.0001)

            # Store segment results
            segment_result = {
                "segment": segment_idx + 1,
                "paired_movements": len(y_direction_pairs),
                "agreement_ratio": agreement_ratio,
                "correlation": correlation,
                "cameras_agree": agreement_ratio > 0.5,
                "y_dominance": y_dominance
            }

            results.append(segment_result)

            print(f"[CROSS-CAM ANALYSIS] Segment {segment_idx + 1}: {len(y_direction_pairs)} paired movements")
            print(f"  Agreement ratio: {agreement_ratio:.2f}, Correlation: {correlation:.2f}")
            print(f"  Y-dominance: {y_dominance:.2f} (higher means clearer Y-axis movement)")

        # No valid results
        if not results:
            print("[CROSS-CAM ANALYSIS] No valid movement analysis could be performed")
            print("[CROSS-CAM ANALYSIS] Using default directions (front=1, side=1)")

            self.front_y_direction = 1
            self.side_y_direction = 1
            self.calibration_active = False
            self.calibration_complete = True
            self.y_verification_state = "Cross-camera analysis: insufficient movement data"

            return False

        # Visualize the cross-camera analysis
        self.visualize_cross_camera_analysis(triple_segments, results)

        # Weight results by Y-dominance and number of movements
        weighted_results = []
        for result in results:
            # Weight by both number of movements and how Y-dominant the movement is
            weight = result["paired_movements"] * min(result["y_dominance"], 5.0)
            weighted_vote = 1 if result["cameras_agree"] else -1
            weighted_results.append((weighted_vote, weight))

        # Calculate final weighted decision
        total_weight = sum(weight for _, weight in weighted_results)
        if total_weight > 0:
            weighted_sum = sum(vote * weight for vote, weight in weighted_results)
            cameras_agree = weighted_sum > 0
        else:
            # Default if weights are zero
            cameras_agree = True

        # Set camera directions based on cross-camera analysis
        if cameras_agree:
            self.front_y_direction = 1
            self.side_y_direction = 1
            decision_msg = "Cameras AGREE on Y direction"
        else:
            self.front_y_direction = 1
            self.side_y_direction = -1
            decision_msg = "Cameras DISAGREE on Y direction - side camera flipped"

        print(f"\n[CROSS-CAM ANALYSIS] FINAL RESULT: {decision_msg}")
        print(
            f"[CROSS-CAM ANALYSIS] Setting front_y_direction={self.front_y_direction}, side_y_direction={self.side_y_direction}")

        # Mark calibration as complete
        self.calibration_active = False
        self.calibration_complete = True

        # Calculate average metrics across all segments
        avg_agreement = sum(r["agreement_ratio"] for r in results) / len(results)
        avg_correlation = sum(r["correlation"] for r in results) / len(results)
        avg_dominance = sum(r["y_dominance"] for r in results) / len(results)

        # Store calibration results
        self.calibration_results = {
            "success": True,
            "method": "cross_camera_analysis",
            "agreement_percentage": avg_agreement,
            "correlation": avg_correlation,
            "y_dominance": avg_dominance,
            "front_direction": self.front_y_direction,
            "side_direction": self.side_y_direction,
            "segments_analyzed": len(results)
        }

        self.y_verification_state = f"Cross-camera analysis: {decision_msg}"

        print(f"[CROSS-CAM ANALYSIS] Analysis complete.")
        print(f"  Average agreement: {avg_agreement:.2f}")
        print(f"  Average correlation: {avg_correlation:.2f}")
        print(f"  Average Y-dominance: {avg_dominance:.2f}")

        return True

    def visualize_cross_camera_analysis(self, triple_segments, results):
        """
        Create an enhanced visualization of the cross-camera analysis

        Args:
            triple_segments: List of (front_segment, side_segment, floor_segment) tuples
            results: Analysis results for each segment
        """
        try:
            # Create figure
            fig = plt.figure(figsize=(15, 12))

            # Plot 1: Y-coordinate movement in all three cameras
            ax1 = fig.add_subplot(221)

            # Prepare data arrays for each camera
            segment_frames = []
            front_y_values = []
            side_x_values = []  # X in side view corresponds to Y in 3D
            floor_xz_values = []  # Floor camera shows X-Z plane

            # Collect data from all segments
            for segment_idx, (front_segment, side_segment, floor_segment) in enumerate(triple_segments):
                segment_result = next((r for r in results if r["segment"] == segment_idx + 1), None)
                if not segment_result:
                    continue

                # Get frames for this segment
                frames = [frame for frame, _, _, _ in front_segment]
                segment_frames.extend(frames)

                # Get Y values from front camera
                front_y = [y for _, _, y, _ in front_segment]
                front_y_values.extend(front_y)

                # Get X values from side camera (corresponds to Y in 3D)
                side_x = [x for _, x, _, _ in side_segment]
                side_x_values.extend(side_x)

                # Get X and Y (Z in 3D) from floor camera
                floor_x = [x for _, x, _, _ in floor_segment]
                floor_y = [y for _, _, y, _ in floor_segment]
                floor_xz_values.extend(zip(floor_x, floor_y))

            # Plot data from each camera with appropriate labels
            if front_y_values and side_x_values:
                ax1.plot(segment_frames, front_y_values, 'g-', linewidth=2, label='Front Camera (Y)')
                ax1.plot(segment_frames, side_x_values, 'm-', linewidth=2, label='Side Camera (X→Y in 3D)')

                # Add segment boundaries
                current_frame = 0
                for segment_idx, (front_segment, _, _) in enumerate(triple_segments):
                    segment_result = next((r for r in results if r["segment"] == segment_idx + 1), None)
                    if not segment_result:
                        continue

                    start_frame = front_segment[0][0]
                    end_frame = front_segment[-1][0]

                    # Highlight segment with color based on agreement
                    color = 'green' if segment_result["cameras_agree"] else 'red'
                    alpha = 0.2
                    ax1.axvspan(start_frame, end_frame, color=color, alpha=alpha)

                    # Add segment label
                    mid_frame = (start_frame + end_frame) // 2
                    ax1.annotate(f"S{segment_idx + 1}", xy=(mid_frame, 0.95), xytext=(mid_frame, 0.95),
                                 ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.6),
                                 color='white', fontsize=9)

            ax1.set_xlabel('Frame Number')
            ax1.set_ylabel('Normalized Position (0-1)')
            ax1.set_title('Y-Coordinate Movement Across Cameras')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')

            # Plot 2: Side-by-side movement analysis
            ax2 = fig.add_subplot(222)

            # For each segment, show key metrics
            if results:
                x = np.arange(len(results))
                width = 0.25

                # Extract metrics
                segments = [r["segment"] for r in results]
                agreements = [r["agreement_ratio"] for r in results]
                correlations = [r["correlation"] for r in results]
                y_dominance = [min(r["y_dominance"], 3.0) / 3.0 for r in results]  # Normalize to 0-1 range

                # Plot bars for each metric
                ax2.bar(x - width, agreements, width, label='Agreement Ratio', color='blue', alpha=0.7)
                ax2.bar(x, correlations, width, label='Correlation', color='green', alpha=0.7)
                ax2.bar(x + width, y_dominance, width, label='Y-Dominance (scaled)', color='orange', alpha=0.7)

                # Decision threshold line
                ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')

                # Annotate with number of movements
                for i, result in enumerate(results):
                    ax2.annotate(f"{result['paired_movements']}",
                                 xy=(i, 1.05),
                                 ha='center', va='bottom',
                                 fontsize=9)

                # Set x-axis labels
                ax2.set_xticks(x)
                ax2.set_xticklabels([f'S{s}' for s in segments])
                ax2.set_ylim(0, 1.1)
                ax2.set_ylabel('Score (0-1)')
                ax2.set_xlabel('Analysis Segments')
                ax2.set_title('Y-Direction Analysis by Segment')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # Plot 3: Floor camera view (X-Z plane)
            ax3 = fig.add_subplot(223)

            if floor_xz_values:
                floor_x = [x for x, _ in floor_xz_values]
                floor_z = [z for _, z in floor_xz_values]

                # Create colormap to show progression
                norm = plt.Normalize(0, len(floor_x))
                colors = plt.cm.viridis(norm(range(len(floor_x))))

                # Plot trajectory with colored points
                for i in range(len(floor_x)):
                    ax3.scatter(floor_x[i], floor_z[i], color=colors[i], s=30, alpha=0.7)

                # Connect points with thin line
                ax3.plot(floor_x, floor_z, 'k-', alpha=0.3, linewidth=1)

            ax3.set_xlabel('X Position (Floor Camera)')
            ax3.set_ylabel('Z Position (Floor Camera)')
            ax3.set_title('Movement in X-Z Plane (Floor Camera View)')
            ax3.grid(True, alpha=0.3)

            # Plot 4: Front vs Side camera correlation
            ax4 = fig.add_subplot(224)

            if len(front_y_values) == len(side_x_values) and len(front_y_values) > 0:
                # Calculate movement
                front_diffs = [y2 - y1 for y1, y2 in zip(front_y_values[:-1], front_y_values[1:])]
                side_diffs = [x2 - x1 for x1, x2 in zip(side_x_values[:-1], side_x_values[1:])]

                # Filter for significant movements
                threshold = self.config.get('y_movement_min_threshold', 0.01)
                sig_moves = [(f, s) for f, s in zip(front_diffs, side_diffs)
                             if abs(f) > threshold and abs(s) > threshold]

                if sig_moves:
                    front_moves = [f for f, _ in sig_moves]
                    side_moves = [s for _, s in sig_moves]

                    # Create scatter plot
                    ax4.scatter(front_moves, side_moves, alpha=0.7, s=30)

                    # Add regression line if possible
                    try:
                        # Calculate regression line
                        z = np.polyfit(front_moves, side_moves, 1)
                        p = np.poly1d(z)

                        # Generate points for the line
                        x_min, x_max = min(front_moves), max(front_moves)
                        x_range = np.linspace(x_min, x_max, 100)

                        # Plot regression line
                        ax4.plot(x_range, p(x_range), 'r--', alpha=0.7)

                        # Add slope to plot
                        slope = z[0]
                        if slope < 0:
                            relationship = "INVERSE"
                            color = 'red'
                        else:
                            relationship = "DIRECT"
                            color = 'green'

                        ax4.text(0.05, 0.95, f"Relationship: {relationship}\nSlope: {slope:.2f}",
                                 transform=ax4.transAxes, fontsize=10,
                                 bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.6),
                                 color='white')
                    except:
                        pass

                    # Add quadrant lines
                    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)

                    # Label quadrants
                    ax4.text(0.25, 0.25, "Same\nDirection", ha='center', va='center',
                             transform=ax4.transAxes, fontsize=9, alpha=0.7)
                    ax4.text(0.75, 0.75, "Same\nDirection", ha='center', va='center',
                             transform=ax4.transAxes, fontsize=9, alpha=0.7)
                    ax4.text(0.25, 0.75, "Opposite\nDirection", ha='center', va='center',
                             transform=ax4.transAxes, fontsize=9, alpha=0.7, color='red')
                    ax4.text(0.75, 0.25, "Opposite\nDirection", ha='center', va='center',
                             transform=ax4.transAxes, fontsize=9, alpha=0.7, color='red')

            ax4.set_xlabel('Front Camera Y Movement')
            ax4.set_ylabel('Side Camera X Movement')
            ax4.set_title('Y-Direction Correlation Analysis')
            ax4.grid(True, alpha=0.3)

            # Final decision box
            # Determine if cameras agree based on weighted results
            agree_segments = sum(1 for r in results if r["cameras_agree"])
            disagree_segments = len(results) - agree_segments

            if agree_segments > disagree_segments:
                decision = "AGREE (Same Direction)"
                direction_text = "front_y_direction=1, side_y_direction=1"
                color = 'green'
            else:
                decision = "DISAGREE (Opposite Direction)"
                direction_text = "front_y_direction=1, side_y_direction=-1"
                color = 'red'

            fig.text(0.5, 0.01, f"FINAL Y-DIRECTION ANALYSIS: Cameras {decision}\n{direction_text}",
                     ha='center', va='bottom', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", fc=color, alpha=0.2))

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            # Save the visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"cross_camera_analysis_{timestamp}.png"
            plt.savefig(output_file, dpi=300)
            print(f"[CROSS-CAM ANALYSIS] Saved analysis visualization to {output_file}")

            # Show plot if interactive
            if plt.isinteractive():
                plt.pause(0.1)
            else:
                plt.close(fig)

        except Exception as e:
            print(f"[CROSS-CAM ANALYSIS] Error creating visualization: {e}")
            import traceback
            traceback.print_exc()

    def find_common_segments(self, front_positions, side_positions):
        """
        Find segments where both cameras detected the object continuously.

        Args:
            front_positions: List of (frame_num, y_value) for front camera
            side_positions: List of (frame_num, y_value) for side camera

        Returns:
            list: List of (front_segment, side_segment) tuples
        """
        # Convert positions to frame-indexed dictionaries for faster lookup
        front_dict = {frame: y for frame, y in front_positions}
        side_dict = {frame: y for frame, y in side_positions}

        # Find all frames where both cameras detected the object
        common_frames = sorted(set(front_dict.keys()) & set(side_dict.keys()))

        if not common_frames:
            return []

        # Split into continuous segments
        segments = []
        current_segment = []
        prev_frame = None

        for frame in common_frames:
            if prev_frame is None or frame == prev_frame + 1:
                # Continuous frame
                current_segment.append(frame)
            else:
                # Gap detected, start a new segment if current one is long enough
                if len(current_segment) >= 5:  # Minimum 5 frames per segment
                    segments.append(current_segment)
                current_segment = [frame]
            prev_frame = frame

        # Add the last segment if it's long enough
        if len(current_segment) >= 5:
            segments.append(current_segment)

        # Convert segments to pairs of (front_data, side_data)
        segment_pairs = []
        for segment in segments:
            front_segment = [(frame, front_dict[frame]) for frame in segment]
            side_segment = [(frame, side_dict[frame]) for frame in segment]
            segment_pairs.append((front_segment, side_segment))

        return segment_pairs

    def analyze_front_side_segments(self, common_segments):
        """
        Fallback method to analyze front and side camera segments when
        three-camera analysis isn't possible.

        Args:
            common_segments: List of (front_segment, side_segment) tuples

        Returns:
            bool: True if analysis successful, False otherwise
        """
        print("[FALLBACK ANALYSIS] Analyzing Y-direction using front and side cameras only...")

        # Results storage
        results = []

        for segment_idx, (front_segment, side_segment) in enumerate(common_segments):
            print(f"[FALLBACK ANALYSIS] Analyzing segment {segment_idx + 1} ({len(front_segment)} frames)")

            # Extract frame-to-frame movements in front camera
            front_y = [y for _, y in front_segment]
            front_diffs = [y2 - y1 for y1, y2 in zip(front_y[:-1], front_y[1:])]

            # Extract frame-to-frame movements in side camera (X is Y in 3D)
            side_x = [x for _, x in side_segment]
            side_diffs = [x2 - x1 for x1, x2 in zip(side_x[:-1], side_x[1:])]

            # Filter for significant movements
            threshold = self.config.get('y_movement_min_threshold', 0.01)
            sig_movements = [(f, s) for f, s in zip(front_diffs, side_diffs)
                             if abs(f) > threshold and abs(s) > threshold]

            if len(sig_movements) < 3:
                print(f"[FALLBACK ANALYSIS] Segment {segment_idx + 1}: Not enough significant movement")
                continue

            # Calculate agreement ratio
            agreement_count = sum(1 for f, s in sig_movements if (f * s) > 0)
            disagreement_count = sum(1 for f, s in sig_movements if (f * s) < 0)

            total_count = agreement_count + disagreement_count
            if total_count > 0:
                agreement_ratio = agreement_count / total_count
            else:
                agreement_ratio = 0.5  # Neutral if no clear movements

            # Calculate correlation coefficient
            try:
                front_np = np.array([f for f, _ in sig_movements])
                side_np = np.array([s for _, s in sig_movements])
                correlation = np.corrcoef(front_np, side_np)[0, 1]
            except:
                correlation = 0

            results.append({
                "segment": segment_idx + 1,
                "movements": len(sig_movements),
                "agreement_ratio": agreement_ratio,
                "correlation": correlation,
                "cameras_agree": agreement_ratio > 0.5
            })

            print(f"[FALLBACK ANALYSIS] Segment {segment_idx + 1}: {len(sig_movements)} movements, "
                  f"Agreement: {agreement_ratio:.2f}, Correlation: {correlation:.2f}")

        # No valid results
        if not results:
            print("[FALLBACK ANALYSIS] No valid movement analysis could be performed")
            print("[FALLBACK ANALYSIS] Using default directions (front=1, side=1)")

            self.front_y_direction = 1
            self.side_y_direction = 1
            self.calibration_active = False
            self.calibration_complete = True
            self.y_verification_state = "Front-side analysis: insufficient movement data"

            return False

        # Weight results by number of movements
        weighted_results = []
        for result in results:
            weight = result["movements"]
            weighted_vote = 1 if result["cameras_agree"] else -1
            weighted_results.append((weighted_vote, weight))

        # Calculate final weighted decision
        total_weight = sum(weight for _, weight in weighted_results)
        if total_weight > 0:
            weighted_sum = sum(vote * weight for vote, weight in weighted_results)
            cameras_agree = weighted_sum > 0
        else:
            # Default if weights are zero
            cameras_agree = True

        # Set camera directions
        if cameras_agree:
            self.front_y_direction = 1
            self.side_y_direction = 1
            decision_msg = "Cameras AGREE on Y direction"
        else:
            self.front_y_direction = 1
            self.side_y_direction = -1
            decision_msg = "Cameras DISAGREE on Y direction - side camera flipped"

        print(f"\n[FALLBACK ANALYSIS] FINAL RESULT: {decision_msg}")
        print(
            f"[FALLBACK ANALYSIS] Setting front_y_direction={self.front_y_direction}, side_y_direction={self.side_y_direction}")

        # Mark calibration as complete
        self.calibration_active = False
        self.calibration_complete = True

        # Calculate average metrics
        avg_agreement = sum(r["agreement_ratio"] for r in results) / len(results)
        avg_correlation = sum(r["correlation"] for r in results) / len(results)

        # Store calibration results
        self.calibration_results = {
            "success": True,
            "method": "front_side_analysis",
            "agreement_percentage": avg_agreement,
            "correlation": avg_correlation,
            "front_direction": self.front_y_direction,
            "side_direction": self.side_y_direction,
            "segments_analyzed": len(results)
        }

        self.y_verification_state = f"Front-side analysis: {decision_msg}"

        return True

    def calibrate_cameras(self, front_point, side_point):
        """
        Calibrate camera directions by analyzing Y-coordinate movements

        Args:
            front_point: (x,y) from front camera or None
            side_point: (x,y) from side camera or None

        Returns:
            bool: True if calibration is complete, False if still calibrating
        """
        # Skip if either point is missing
        if front_point is None or side_point is None:
            return False

        # Increment calibration frame counter
        self.calibration_frame_count += 1

        # Extract Y values with normalization (0-1 range)
        front_y_norm = front_point[1] / self.height_front
        # Side camera: X coordinate in side view is Y in 3D
        side_y_norm = side_point[0] / self.width_side

        # Store raw values
        self.front_y_raw.append(front_y_norm)
        self.side_y_raw.append(side_y_norm)

        # Calculate movements if we have at least 2 points
        if len(self.front_y_raw) >= 2:
            front_movement = self.front_y_raw[-1] - self.front_y_raw[-2]
            side_movement = self.side_y_raw[-1] - self.side_y_raw[-2]

            # Only register significant movements to avoid noise
            threshold = self.config.get('y_movement_min_threshold', 0.01)
            if abs(front_movement) > threshold and abs(side_movement) > threshold:
                self.front_y_movements.append(front_movement)
                self.side_y_movements.append(side_movement)

        # Update visualization during calibration
        self.update_calibration_visualization()

        # Check if we have enough data to complete calibration
        if self.calibration_frame_count >= self.config['y_calibration_frames'] and len(self.front_y_movements) >= 3:
            self.complete_calibration()
            return True

        return False

    def complete_calibration(self):
        """Complete the camera calibration process and determine Y directions"""
        try:
            # Only use significant movements for direction analysis
            threshold = self.config.get('y_movement_min_threshold', 0.01)
            significant_pairs = [(f, s) for f, s in zip(self.front_y_movements, self.side_y_movements)
                                 if abs(f) > threshold and abs(s) > threshold]

            if len(significant_pairs) < 3:
                print("[CALIB] Not enough significant movement pairs for calibration")
                self.calibration_results = {
                    "success": False,
                    "reason": "Not enough significant movements",
                    "method": "standard"
                }
                # Use default directions
                self.front_y_direction = 1
                self.side_y_direction = 1
            else:
                # Calculate how often front and side camera movements agree in direction
                agreement_count = sum(1 for f, s in significant_pairs if (f * s) > 0)
                disagreement_count = sum(1 for f, s in significant_pairs if (f * s) < 0)

                total_count = agreement_count + disagreement_count
                agreement_percentage = agreement_count / total_count if total_count > 0 else 0.5

                # Calculate correlation between front and side movements
                try:
                    front_moves = np.array([f for f, _ in significant_pairs])
                    side_moves = np.array([s for _, s in significant_pairs])
                    correlation = np.corrcoef(front_moves, side_moves)[0, 1]
                except:
                    correlation = 0

                print(f"[CALIB] Agreement: {agreement_percentage:.2f}, Correlation: {correlation:.2f}")
                print(f"[CALIB] {agreement_count} agreements, {disagreement_count} disagreements")

                # Determine camera directions based on agreement
                if agreement_percentage >= 0.5:
                    # Cameras see movement in same direction
                    print("[CALIB] Cameras AGREE on movement direction")
                    self.front_y_direction = 1
                    self.side_y_direction = 1
                else:
                    # Cameras see movement in opposite directions - flip side camera
                    print("[CALIB] Cameras DISAGREE on movement direction")
                    self.front_y_direction = 1
                    self.side_y_direction = -1

                # Store calibration results
                self.calibration_results = {
                    "success": True,
                    "agreement_percentage": agreement_percentage,
                    "correlation": correlation,
                    "movements_analyzed": len(significant_pairs),
                    "method": "standard"
                }

            # Mark calibration as complete
            self.calibration_active = False
            self.calibration_complete = True
            print(f"[CALIB] Calibration complete. Front Y direction: {self.front_y_direction}, "
                  f"Side Y direction: {self.side_y_direction}")

            # Set initial verification state
            self.y_verification_state = "Calibrated. Y-verification active."

        except Exception as e:
            print(f"[CALIB] Error completing calibration: {e}")
            import traceback
            traceback.print_exc()

            # Use default settings
            self.calibration_active = False
            self.calibration_complete = True
            self.front_y_direction = 1
            self.side_y_direction = 1
            self.y_verification_state = "Calibration error. Using defaults."
            self.calibration_results = {
                "success": False,
                "reason": f"Error: {str(e)}",
                "method": "standard"
            }

    def verify_y_coordinate_directions(self, front_point, side_point):
        """
        Verify the consistency of Y-coordinate readings from front and side cameras

        Args:
            front_point: (x,y) from front camera or None
            side_point: (x,y) from side camera or None

        Returns:
            tuple: (front_y, side_y, agreement_score, correlation)
        """
        if front_point is None or side_point is None:
            return None, None, 0, 0

        # Extract and normalize Y coordinates
        front_y_norm = front_point[1] / self.height_front
        side_y_norm = side_point[0] / self.width_side  # Side camera X is Y in 3D

        # Apply calibrated directions
        front_y = front_y_norm * self.front_y_direction
        side_y = side_y_norm * self.side_y_direction

        # Store Y values for history
        self.front_y_values.append(front_y)
        self.side_y_values.append(side_y)

        # Calculate agreement score
        agreement = 0
        correlation = 0

        # Get validation window size
        window_size = min(len(self.front_y_values), self.config['y_validation_window'])

        if window_size >= 3:  # Need at least 3 points for reliable validation
            # Get recent values within window
            recent_front_y = self.front_y_values[-window_size:]
            recent_side_y = self.side_y_values[-window_size:]

            # Calculate movements within window
            front_movements = [recent_front_y[i + 1] - recent_front_y[i] for i in range(window_size - 1)]
            side_movements = [recent_side_y[i + 1] - recent_side_y[i] for i in range(window_size - 1)]

            # Find significant movements (filter out noise)
            threshold = self.config.get('y_movement_min_threshold', 0.01)
            significant_moves = [(f, s) for f, s in zip(front_movements, side_movements)
                                 if abs(f) > threshold and abs(s) > threshold]

            if significant_moves:
                # Calculate agreement score based on movement sign matching
                agreement_count = sum(1 for f, s in significant_moves if (f > 0 and s > 0) or (f < 0 and s < 0))
                agreement = agreement_count / len(significant_moves)

                # Calculate correlation between movements using specified method
                try:
                    corr_method = self.config['y_correlation_method']
                    front_vals = np.array([f for f, _ in significant_moves])
                    side_vals = np.array([s for _, s in significant_moves])

                    if corr_method == 'pearson':
                        correlation = np.corrcoef(front_vals, side_vals)[0, 1]
                    elif corr_method == 'spearman':
                        from scipy.stats import spearmanr
                        correlation, _ = spearmanr(front_vals, side_vals)
                    elif corr_method == 'kendall':
                        from scipy.stats import kendalltau
                        correlation, _ = kendalltau(front_vals, side_vals)
                    else:
                        # Default to Pearson
                        correlation = np.corrcoef(front_vals, side_vals)[0, 1]

                    # Handle NaN correlation
                    if np.isnan(correlation):
                        correlation = 0
                except Exception as e:
                    print(f"Error calculating correlation: {e}")
                    correlation = 0

                # Store correlation score
                self.y_correlation_scores.append(correlation)

                # Detect if this is a conflict frame
                if agreement < self.config['y_disagreement_threshold']:
                    self.y_conflict_frames.append(len(self.front_y_values) - 1)  # Current frame index

            # No significant movements in window
            else:
                # If no significant movement, use the most recent agreement score
                if self.y_agreement_scores:
                    agreement = self.y_agreement_scores[-1]
                else:
                    agreement = 1.0  # Default to agreement if no history

                # Use previous correlation or 1.0 if no history
                if self.y_correlation_scores:
                    correlation = self.y_correlation_scores[-1]
                else:
                    correlation = 1.0

        # Store agreement score
        self.y_agreement_scores.append(agreement)

        # Update verification state based on latest data
        if agreement >= 0.8:
            self.y_verification_state = "Cameras agree on Y-direction"
        elif agreement >= 0.5:
            self.y_verification_state = "Partial agreement on Y-direction"
        else:
            self.y_verification_state = "Cameras disagree on Y-direction"

        # Additional detail when correlations are available
        if correlation != 0 and len(self.y_correlation_scores) > 0:
            avg_correlation = sum(self.y_correlation_scores[-10:]) / min(len(self.y_correlation_scores), 10)
            if avg_correlation < 0:
                self.y_verification_state += " (inverse correlation)"

        return front_y, side_y, agreement, correlation

    def reconstruct_3d_point(self, front_point, side_point, floor_point, confidences):
        """
        Reconstruct a 3D point using exclusive camera assignments:
        - Floor camera: X and Z coordinates
        - Front/Side cameras: Y coordinate (with verification)

        Args:
            front_point: (x,y) from front camera or None
            side_point: (x,y) from side camera or None
            floor_point: (x,y) from floor camera or None
            confidences: (front_conf, side_conf, floor_conf)

        Returns:
            tuple: (x, y, z) 3D coordinates or None if reconstruction fails
        """
        if floor_point is None:
            # Need floor camera for X and Z coordinates
            return None

        # Get confidences
        front_conf, side_conf, floor_conf = confidences

        # FLOOR CAMERA: X and Z coordinates (with normalization to 0-1 range)
        floor_x_norm = floor_point[0] / self.width_floor
        floor_z_norm = floor_point[1] / self.height_floor  # Y in floor image is Z in 3D

        # Scale to display range
        x = floor_x_norm * self.config['display_scale']
        z = floor_z_norm * self.config['display_scale']

        # FRONT/SIDE CAMERAS: Y coordinate with verification and blending
        y = None

        # Try to get Y coordinate from both front and side cameras with proper blending
        front_y, side_y = None, None

        if front_point is not None:
            # Normalize and apply direction
            front_y_norm = front_point[1] / self.height_front
            front_y = front_y_norm * self.front_y_direction * self.config['display_scale']

        if side_point is not None:
            # Normalize and apply direction (X in side view is Y in 3D)
            side_y_norm = side_point[0] / self.width_side
            side_y = side_y_norm * self.side_y_direction * self.config['display_scale']

        # Choose Y-coordinate blending method based on configuration
        blend_method = self.config.get('y_blending_method', 'adaptive')

        if front_y is not None and side_y is not None:
            # Both cameras have valid Y coordinates

            # Check for conflicts
            y_diff = abs(front_y - side_y) / self.config['display_scale']  # Normalize diff
            conflict = y_diff > self.config['y_disagreement_threshold']

            if conflict and self.config['highlight_y_conflicts']:
                # Mark as conflict frame
                if len(self.front_y_values) - 1 not in self.y_conflict_frames:
                    self.y_conflict_frames.append(len(self.front_y_values) - 1)

            # Apply selected blending method
            if blend_method == 'weighted':
                # Simple weighted average based on confidence
                total_conf = front_conf + side_conf
                if total_conf > 0:
                    y = (front_y * front_conf + side_y * side_conf) / total_conf
                else:
                    y = (front_y + side_y) / 2  # Equal weights if no confidence

            elif blend_method == 'best_confidence':
                # Use the camera with higher confidence
                if front_conf >= side_conf:
                    y = front_y
                else:
                    y = side_y

            elif blend_method == 'adaptive':
                # Adaptive blending based on agreement history
                if self.y_agreement_scores and len(self.y_agreement_scores) > 10:
                    # Use recent agreement trend
                    recent_agreement = sum(self.y_agreement_scores[-10:]) / 10

                    if conflict:
                        # Handle conflict based on configuration
                        resolution = self.config.get('y_conflict_resolution', 'voting')

                        if resolution == 'voting':
                            # Use camera that historically agrees better with overall trend
                            if recent_agreement >= 0.5:
                                # Cameras mostly agree, use confidence-weighted value
                                total_conf = front_conf + side_conf
                                if total_conf > 0:
                                    y = (front_y * front_conf + side_y * side_conf) / total_conf
                                else:
                                    y = (front_y + side_y) / 2
                            else:
                                # Cameras historically disagree
                                # Use camera with higher confidence
                                if front_conf >= side_conf:
                                    y = front_y
                                else:
                                    y = side_y

                        elif resolution == 'highest_confidence':
                            # Always use highest confidence camera for conflicts
                            if front_conf >= side_conf:
                                y = front_y
                            else:
                                y = side_y

                        else:  # 'most_recent' or default
                            # Use most recent trend to decide
                            if len(self.y_agreement_scores) > 20:
                                recent_trend = sum(self.y_agreement_scores[-10:]) / 10
                                older_trend = sum(self.y_agreement_scores[-20:-10]) / 10

                                improving = recent_trend > older_trend
                                if improving:
                                    # Trend is improving, use weighted average
                                    total_conf = front_conf + side_conf
                                    if total_conf > 0:
                                        y = (front_y * front_conf + side_y * side_conf) / total_conf
                                    else:
                                        y = (front_y + side_y) / 2
                                else:
                                    # Trend is not improving, use highest confidence
                                    if front_conf >= side_conf:
                                        y = front_y
                                    else:
                                        y = side_y
                            else:
                                # Not enough trend data, use confidence
                                if front_conf >= side_conf:
                                    y = front_y
                                else:
                                    y = side_y
                    else:
                        # No conflict, use weighted average
                        total_conf = front_conf + side_conf
                        if total_conf > 0:
                            y = (front_y * front_conf + side_y * side_conf) / total_conf
                        else:
                            y = (front_y + side_y) / 2
                else:
                    # Not enough history, use simple weighted average
                    total_conf = front_conf + side_conf
                    if total_conf > 0:
                        y = (front_y * front_conf + side_y * side_conf) / total_conf
                    else:
                        y = (front_y + side_y) / 2

            else:  # 'average' or any other value
                # Simple average
                y = (front_y + side_y) / 2

        elif front_y is not None:
            # Only front camera has Y coordinate
            y = front_y

        elif side_y is not None:
            # Only side camera has Y coordinate
            y = side_y

        else:
            # No Y coordinate available from either camera
            if self.trajectory_3d:
                # Use previous Y value if available
                y = self.trajectory_3d[-1][1]
            else:
                # No previous Y, use middle of display range
                y = self.config['display_scale'] / 2

        # Create 3D point with exclusive camera assignments
        point_3d = (x, y, z)

        # Apply smoothing if enabled and we have previous positions
        smoothing = self.config.get('smoothing_factor', 0)
        if smoothing > 0 and self.trajectory_3d:
            prev_point = self.trajectory_3d[-1]
            smoothed_point = (
                prev_point[0] * smoothing + point_3d[0] * (1 - smoothing),
                prev_point[1] * smoothing + point_3d[1] * (1 - smoothing),
                prev_point[2] * smoothing + point_3d[2] * (1 - smoothing)
            )
            return smoothed_point

        return point_3d

    def update_calibration_visualization(self):
        """Update visualization during camera calibration phase"""
        try:
            # Update Y-coordinate verification plot to show calibration data
            if self.front_y_raw and self.side_y_raw:
                # Clear previous data
                self.line_front_y.set_data([], [])
                self.line_side_y.set_data([], [])
                self.line_y_agreement.set_data([], [])

                # Plot raw Y values from both cameras
                frames = list(range(len(self.front_y_raw)))

                self.line_front_y.set_data(frames, self.front_y_raw)
                self.line_side_y.set_data(frames, self.side_y_raw)

                # Update Y-verify plot
                self.ax_y_verify.relim()
                self.ax_y_verify.autoscale_view()

                # Update title to show calibration progress
                max_frames = self.config['y_calibration_frames']
                progress = min(100, int(self.calibration_frame_count / max_frames * 100))
                self.ax_y_verify.set_title(f'Y-Calibration: {progress}% complete', color='orange')

                # Add calibration instruction text
                self.ax_y_verify.text(0.05, 0.05, "Move object up and down\nto calibrate Y-direction",
                                      transform=self.ax_y_verify.transAxes, color='blue', fontsize=12,
                                      bbox=dict(facecolor='white', alpha=0.7))

                # Add movement indicators if we have movements
                if self.front_y_movements and self.side_y_movements:
                    # Plot movement directions
                    if len(self.front_y_movements) > 0 and len(self.side_y_movements) > 0:
                        movement_frames = list(range(len(self.front_y_movements)))
                        # Scale movements for visibility
                        scale = 0.1 / max(abs(max(self.front_y_movements)), abs(min(self.front_y_movements)))
                        scaled_front = [0.5 + m * scale for m in self.front_y_movements]
                        scaled_side = [0.5 + m * scale for m in self.side_y_movements]

                        # Plot on top of axes
                        self.ax_movement.clear()
                        self.ax_movement.plot(movement_frames, scaled_front, 'g-', label='Front Moves')
                        self.ax_movement.plot(movement_frames, scaled_side, 'm-', label='Side Moves')

                        # Draw horizontal line at center
                        self.ax_movement.axhline(y=0.5, color='k', linestyle='-', alpha=0.3)

                        # Annotate agreement
                        agreements = [1 if (f > 0 and s > 0) or (f < 0 and s < 0) else -1
                                      for f, s in zip(self.front_y_movements, self.side_y_movements)]

                        for i, agree in enumerate(agreements):
                            if i % 3 == 0:  # Only mark every 3rd point to avoid clutter
                                color = 'green' if agree > 0 else 'red'
                                self.ax_movement.axvline(x=i, color=color, linestyle='--', alpha=0.3)

                        self.ax_movement.set_title('Camera Movement Comparison')
                        self.ax_movement.set_xlabel('Movement Sample')
                        self.ax_movement.set_ylabel('Direction')
                        self.ax_movement.set_ylim(0, 1)
                        self.ax_movement.legend()

            # Update other plots with minimal placeholder data
            x_data = [0, 100]
            y_data = [50, 50]
            z_data = [50, 50]

            # 3D plot placeholder
            self.line_3d.set_data(x_data, y_data)
            self.line_3d.set_3d_properties(z_data)

            # Update 2D plots
            self.line_top.set_data(x_data, z_data)
            self.line_front.set_data(x_data, y_data)
            self.line_side.set_data(z_data, y_data)

            # Set 3D plot with calibration message
            self.ax_3d.clear()
            self.ax_3d.set_xlim(0, 100)
            self.ax_3d.set_ylim(0, 100)
            self.ax_3d.set_zlim(0, 100)
            self.ax_3d.set_xlabel('X (Floor)')
            self.ax_3d.set_ylabel('Y (Front/Side)')
            self.ax_3d.set_zlabel('Z (Floor)')

            # Add calibration message to 3D plot
            max_frames = self.config['y_calibration_frames']
            progress = min(100, int(self.calibration_frame_count / max_frames * 100))

            self.ax_3d.text(50, 50, 60, f"CAMERA CALIBRATION\n{progress}% Complete",
                            fontsize=16, color='red', ha='center', weight='bold')
            self.ax_3d.text(50, 50, 40, "Move the object up and down\nto calibrate Y coordinates",
                            fontsize=12, color='blue', ha='center')

            # Add calibration instructions to 2D plots
            for ax in [self.ax_top, self.ax_front, self.ax_side]:
                ax.clear()
                ax.text(0.5, 0.5, "Calibrating...", ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='red')

            # Update the figure
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

        except Exception as e:
            print(f"Error in update_calibration_visualization: {e}")
            import traceback
            traceback.print_exc()

    def create_y_comparison_visualization(self, front_point, side_point, front_frame, side_frame):
        """
        Create a visualization showing the Y-coordinate comparison between cameras

        Args:
            front_point: (x,y) from front camera or None
            side_point: (x,y) from side camera or None
            front_frame: Front camera frame
            side_frame: Side camera frame

        Returns:
            numpy.ndarray: Comparison visualization frame
        """
        # Create a blank canvas
        h, w = 600, 800
        vis = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Define regions
        graph_region = (10, 10, 780, 240)  # x, y, width, height
        front_region = (10, 260, 380, 280)
        side_region = (410, 260, 380, 280)

        # Draw borders around regions
        cv2.rectangle(vis, (graph_region[0], graph_region[1]),
                      (graph_region[0] + graph_region[2], graph_region[1] + graph_region[3]),
                      (200, 200, 200), 2)
        cv2.rectangle(vis, (front_region[0], front_region[1]),
                      (front_region[0] + front_region[2], front_region[1] + front_region[3]),
                      (0, 255, 0), 2)
        cv2.rectangle(vis, (side_region[0], side_region[1]),
                      (side_region[0] + side_region[2], side_region[1] + side_region[3]),
                      (255, 0, 255), 2)

        # Add titles
        cv2.putText(vis, "Y-Coordinate Comparison", (graph_region[0] + 200, graph_region[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(vis, "Front Camera", (front_region[0] + 100, front_region[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        cv2.putText(vis, "Side Camera", (side_region[0] + 100, side_region[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 150), 2)

        # Plot the Y-coordinate graph
        if self.front_y_values and self.side_y_values:
            # Graph dimensions
            gx, gy, gw, gh = graph_region

            # Draw coordinate grid
            for i in range(5):
                y = gy + int(gh * i / 4)
                cv2.line(vis, (gx, y), (gx + gw, y), (230, 230, 230), 1)
                value = 1.0 - i / 4
                cv2.putText(vis, f"{value:.1f}", (gx - 25, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Draw time grid
            num_frames = min(50, len(self.front_y_values))
            for i in range(6):
                x = gx + int(gw * i / 5)
                cv2.line(vis, (x, gy), (x, gy + gh), (230, 230, 230), 1)
                frame = len(self.front_y_values) - num_frames + int(num_frames * i / 5)
                if frame >= 0 and frame < len(self.front_y_values):
                    cv2.putText(vis, f"{frame}", (x - 10, gy + gh + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Draw y-axis label
            cv2.putText(vis, "Y Value", (gx - 55, gy + int(gh / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw x-axis label
            cv2.putText(vis, "Frame", (gx + int(gw / 2), gy + gh + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Plot front camera values (green)
            values = self.front_y_values[-num_frames:] if len(
                self.front_y_values) > num_frames else self.front_y_values
            scale_x = gw / (num_frames - 1) if num_frames > 1 else 0
            scale_y = gh

            for i in range(1, len(values)):
                x1 = gx + int((i - 1) * scale_x)
                y1 = gy + int((1 - values[i - 1]) * scale_y)
                x2 = gx + int(i * scale_x)
                y2 = gy + int((1 - values[i]) * scale_y)
                cv2.line(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Plot side camera values (magenta)
            values = self.side_y_values[-num_frames:] if len(
                self.side_y_values) > num_frames else self.side_y_values

            for i in range(1, len(values)):
                x1 = gx + int((i - 1) * scale_x)
                y1 = gy + int((1 - values[i - 1]) * scale_y)
                x2 = gx + int(i * scale_x)
                y2 = gy + int((1 - values[i]) * scale_y)
                cv2.line(vis, (x1, y1), (x2, y2), (200, 0, 200), 2)

            # Draw agreement scores in black
            if self.y_agreement_scores:
                values = self.y_agreement_scores[-num_frames:] if len(
                    self.y_agreement_scores) > num_frames else self.y_agreement_scores
                for i in range(1, len(values)):
                    x1 = gx + int((i - 1) * scale_x)
                    y1 = gy + int((1 - values[i - 1]) * scale_y)
                    x2 = gx + int(i * scale_x)
                    y2 = gy + int((1 - values[i]) * scale_y)
                    cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)

            # Mark calibration points or conflicts
            if self.y_conflict_frames:
                for cf in self.y_conflict_frames:
                    if cf >= len(self.front_y_values) - num_frames and cf < len(self.front_y_values):
                        idx = cf - (len(self.front_y_values) - num_frames)
                        x = gx + int(idx * scale_x)
                        cv2.line(vis, (x, gy), (x, gy + gh), (0, 0, 255), 1, cv2.LINE_AA)

        # Display camera frames with Y-coordinate highlighted
        if front_frame is not None:
            # Resize front frame to fit region
            h, w = front_frame.shape[:2]
            scale = min(front_region[2] / w, front_region[3] / h)
            resized = cv2.resize(front_frame, (int(w * scale), int(h * scale)))

            # Place in region
            rx, ry = front_region[0], front_region[1]
            vis[ry:ry + resized.shape[0], rx:rx + resized.shape[1]] = resized

            # Highlight Y-coordinate
            if front_point is not None:
                # Mark the Y-coordinate
                fy_norm = front_point[1] / self.height_front
                y_pos = ry + int(resized.shape[0] * fy_norm)
                cv2.line(vis, (rx, y_pos), (rx + resized.shape[1], y_pos), (0, 255, 0), 2)

                # Add Y value info
                y_val = fy_norm * self.front_y_direction
                cv2.putText(vis, f"Y: {y_val:.2f} (Dir: {self.front_y_direction})",
                            (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if side_frame is not None:
            # Resize side frame to fit region
            h, w = side_frame.shape[:2]
            scale = min(side_region[2] / w, side_region[3] / h)
            resized = cv2.resize(side_frame, (int(w * scale), int(h * scale)))

            # Place in region
            rx, ry = side_region[0], side_region[1]
            vis[ry:ry + resized.shape[0], rx:rx + resized.shape[1]] = resized

            # Highlight Y-coordinate (X in side view is Y in 3D)
            if side_point is not None:
                # Mark the Y-coordinate
                sx_norm = side_point[0] / self.width_side
                x_pos = rx + int(resized.shape[1] * sx_norm)
                cv2.line(vis, (x_pos, ry), (x_pos, ry + resized.shape[0]), (255, 0, 255), 2)

                # Add Y value info
                y_val = sx_norm * self.side_y_direction
                cv2.putText(vis, f"Y: {y_val:.2f} (Dir: {self.side_y_direction})",
                            (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Add calibration/verification status
        y_status_color = (0, 0, 0)
        if "disagree" in self.y_verification_state.lower():
            y_status_color = (0, 0, 255)
        elif "agree" in self.y_verification_state.lower():
            y_status_color = (0, 150, 0)
        elif "calibrat" in self.y_verification_state.lower():
            y_status_color = (255, 165, 0)

        cv2.putText(vis, self.y_verification_state, (20, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, y_status_color, 2)

        # Add correlation info if available
        if self.y_correlation_scores and len(self.y_correlation_scores) > 0:
            corr = self.y_correlation_scores[-1]
            cv2.putText(vis, f"Correlation: {corr:.2f}", (20, 580),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)

        # Add legend
        cv2.line(vis, (600, 550), (620, 550), (0, 200, 0), 2)
        cv2.putText(vis, "Front Y", (625, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.line(vis, (600, 570), (620, 570), (200, 0, 200), 2)
        cv2.putText(vis, "Side Y", (625, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.line(vis, (600, 590), (620, 590), (0, 0, 0), 1)
        cv2.putText(vis, "Agreement", (625, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return vis

    def update_visualization(self):
        """Update all visualization plots with current data"""
        # If we're in calibration mode, show calibration visualization
        if self.calibration_active:
            self.update_calibration_visualization()
            return

        # For normal tracking, need trajectory data
        if not self.trajectory_3d:
            return

        try:
            # Extract x, y, z coordinates
            x_points = [p[0] for p in self.trajectory_3d]
            y_points = [p[1] for p in self.trajectory_3d]
            z_points = [p[2] for p in self.trajectory_3d]

            # Update 3D trajectory plot
            self.line_3d.set_data(x_points, y_points)
            self.line_3d.set_3d_properties(z_points)

            # Update current position point
            current_point = self.trajectory_3d[-1]
            self.point_3d._offsets3d = ([current_point[0]], [current_point[1]], [current_point[2]])

            # Adjust 3D plot limits
            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)
            z_min, z_max = min(z_points), max(z_points)

            # Add some padding
            padding = 5.0
            self.ax_3d.set_xlim(x_min - padding, x_max + padding)
            self.ax_3d.set_ylim(y_min - padding, y_max + padding)
            self.ax_3d.set_zlim(z_min - padding, z_max + padding)

            # Update top view (X-Z plane)
            self.line_top.set_data(x_points, z_points)
            self.point_top.set_offsets(np.column_stack([x_points[-1], z_points[-1]]))
            self.ax_top.relim()
            self.ax_top.autoscale_view()

            # Update front view (X-Y plane)
            self.line_front.set_data(x_points, y_points)
            self.point_front.set_offsets(np.column_stack([x_points[-1], y_points[-1]]))
            self.ax_front.relim()
            self.ax_front.autoscale_view()

            # Update side view (Z-Y plane)
            self.line_side.set_data(z_points, y_points)
            self.point_side.set_offsets(np.column_stack([z_points[-1], y_points[-1]]))
            self.ax_side.relim()
            self.ax_side.autoscale_view()

            # Update dimension movement plot
            frames = list(range(len(self.dimension_movements['X'])))
            self.line_x_movement.set_data(frames, self.dimension_movements['X'])
            self.line_y_movement.set_data(frames, self.dimension_movements['Y'])
            self.line_z_movement.set_data(frames, self.dimension_movements['Z'])
            self.ax_movement.relim()
            self.ax_movement.autoscale_view()

            # Update Y-coordinate verification plot
            if self.front_y_values and self.side_y_values:
                # Get appropriate window size for display
                window_size = min(50, len(self.front_y_values))
                display_start = max(0, len(self.front_y_values) - window_size)

                # Extract display window for front camera values
                display_front_y = self.front_y_values[display_start:]
                frames_front = list(range(display_start, display_start + len(display_front_y)))
                self.line_front_y.set_data(frames_front, display_front_y)

                # Extract display window for side camera values
                display_side_y = self.side_y_values[display_start:]
                frames_side = list(range(display_start, display_start + len(display_side_y)))
                self.line_side_y.set_data(frames_side, display_side_y)

                # Extract display window for agreement scores
                if self.y_agreement_scores:
                    display_agreement = self.y_agreement_scores[display_start:display_start + window_size]
                    frames_agreement = list(range(display_start, display_start + len(display_agreement)))
                    self.line_y_agreement.set_data(frames_agreement, display_agreement)

                # Update axis limits
                self.ax_y_verify.relim()
                self.ax_y_verify.autoscale_view()

                # Update Y-verification title with status
                status_text = self.y_verification_state
                if len(status_text) > 30:
                    status_text = status_text[:30] + "..."

                # Change title color based on status
                if "agree" in self.y_verification_state.lower():
                    self.ax_y_verify.set_title(f'Y-Verification: {status_text}', color='green')
                elif "disagree" in self.y_verification_state.lower():
                    self.ax_y_verify.set_title(f'Y-Verification: {status_text}', color='red')
                else:
                    self.ax_y_verify.set_title(f'Y-Verification: {status_text}')

                # Add Y-correlation value to the title if available
                if self.y_correlation_scores and len(self.y_correlation_scores) > 0:
                    corr = self.y_correlation_scores[-1]
                    self.ax_y_verify.set_title(f'Y-Verification: {status_text} (r={corr:.2f})')

                # Highlight disagreement points if enabled
                if self.config['highlight_y_conflicts'] and self.y_conflict_frames:
                    # Remove previous conflict highlights
                    for artist in self.ax_y_verify.collections:
                        if isinstance(artist, plt.matplotlib.collections.PathCollection):
                            if artist != self.point_3d and artist != self.point_front and artist != self.point_side and artist != self.point_top:
                                artist.remove()

                    # Add new conflict highlights
                    recent_conflicts = [cf for cf in self.y_conflict_frames if cf >= display_start]
                    if recent_conflicts:
                        conflict_x = recent_conflicts
                        conflict_y1 = [self.front_y_values[cf] if cf < len(self.front_y_values) else 0 for cf in
                                       recent_conflicts]
                        conflict_y2 = [self.side_y_values[cf] if cf < len(self.side_y_values) else 0 for cf in
                                       recent_conflicts]

                        # Add conflict markers
                        self.ax_y_verify.scatter(conflict_x, conflict_y1, color='red', marker='x', s=40, alpha=0.7)
                        self.ax_y_verify.scatter(conflict_x, conflict_y2, color='red', marker='x', s=40, alpha=0.7)

            # Remove previous text annotations
            for txt in self.text_annotations:
                if txt in self.ax_3d.texts:
                    txt.remove()
            self.text_annotations = []

            # Add camera assignment info
            info_txt = self.ax_3d.text(x_min, y_min, z_max + padding / 2,
                                       "Floor: X,Z | Front/Side: Y",
                                       color='black', fontsize=10)
            self.text_annotations.append(info_txt)

            # Add primary plane info
            primary_plane = self.detect_primary_plane()
            plane_txt = self.ax_3d.text(x_min, y_min + padding / 2, z_max + padding / 2,
                                        f"Primary plane: {primary_plane}",
                                        color='black', fontsize=10)
            self.text_annotations.append(plane_txt)

            # Add Y-coordinate verification info with enhanced visibility
            if self.calibration_complete:
                if len(self.y_correlation_scores) > 0:
                    corr = self.y_correlation_scores[-1]
                    corr_info = f" (r={corr:.2f})"
                else:
                    corr_info = ""

                # Create more detailed Y-coordinate info
                y_info = (f"Front Y dir: {self.front_y_direction} | "
                          f"Side Y dir: {self.side_y_direction}{corr_info}")

                # Choose text color based on verification state
                if "agree" in self.y_verification_state.lower():
                    y_color = 'green'
                elif "disagree" in self.y_verification_state.lower():
                    y_color = 'red'
                else:
                    y_color = 'blue'

                y_txt = self.ax_3d.text(x_min, y_min + padding, z_max + padding / 2,
                                        y_info, color=y_color, fontsize=9, weight='bold')
                self.text_annotations.append(y_txt)

                # Add verification status
                status_txt = self.ax_3d.text(x_min, y_min + padding * 1.5, z_max + padding / 2,
                                             f"Y Status: {self.y_verification_state}",
                                             color=y_color, fontsize=9)
                self.text_annotations.append(status_txt)

            # Add annotations for start and current positions
            if len(self.trajectory_3d) > 1:
                start_txt = self.ax_3d.text(x_points[0], y_points[0], z_points[0], "Start", color='green')
                current_txt = self.ax_3d.text(x_points[-1], y_points[-1], z_points[-1], "Current", color='blue')
                self.text_annotations.append(start_txt)
                self.text_annotations.append(current_txt)

            # Update the figure
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

        except Exception as e:
            print(f"Error in update_visualization: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main loop for tracking and visualization"""
        try:
            print("Starting tracking with exclusive camera assignments and Y-coordinate verification.")
            print("Press ESC to stop.")

            # Run cross-camera analysis if enabled
            if self.config.get('use_cross_camera_analysis', False):
                print("\nRunning cross-camera Y-direction analysis...")
                frames_to_analyze = self.config.get('analysis_frames', 300)
                sample_interval = self.config.get('analysis_interval', 2)

                try:
                    # Perform 3-camera cross-analysis
                    self.analyze_three_camera_movement(frames_to_analyze, sample_interval)
                    print("\nCross-camera analysis complete. Starting normal tracking.")
                except Exception as e:
                    print(f"Error during cross-camera analysis: {e}")
                    print("Using default Y-direction settings (front=1, side=1)")
                    import traceback
                    traceback.print_exc()

                    # Ensure we exit analysis mode
                    self.calibration_active = False
                    self.calibration_complete = True
                    self.front_y_direction = 1
                    self.side_y_direction = 1

            # Continue with normal tracking loop
            while True:
                try:
                    # Debug info for frame processing
                    if self.frame_count % 30 == 0:  # Print every 30 frames
                        print(f"Processing frame {self.frame_count} (Calibration: {self.calibration_active})")

                    # Process the current frame
                    if not self.process_frame():
                        print("End of video(s) or processing error")
                        break

                    # Check for ESC key to exit
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        print("User stopped tracking")
                        break
                except Exception as e:
                    print(f"Error in run loop: {e}")
                    import traceback
                    traceback.print_exc()
                    # Try to continue with next frame
                    self.frame_count += 1
                    continue

        finally:
            # Clean up resources
            try:
                self.cap_front.release()
                self.cap_side.release()
                self.cap_floor.release()
                cv2.destroyAllWindows()
                plt.close(self.fig)
            except Exception as e:
                print(f"Error during cleanup: {e}")

            print(f"Tracking complete. Processed {self.frame_count} frames.")
            print(f"Tracked {len(self.trajectory_3d)} 3D points.")

    def process_frame(self):
        """Process one frame from each camera and update 3D position"""
        try:
            # Read frames from all cameras
            ret_front, frame_front = self.cap_front.read()
            ret_side, frame_side = self.cap_side.read()
            ret_floor, frame_floor = self.cap_floor.read()

            # Check if we've reached the end of any videos
            if not all([ret_front, ret_side, ret_floor]):
                return False

            # Record timestamp
            self.timestamps.append(time.time())

            # Detect red object in each view with confidence
            try:
                front_result = self.detect_red_object(frame_front, 'front')
                side_result = self.detect_red_object(frame_side, 'side')
                floor_result = self.detect_red_object(frame_floor, 'floor')

                # Unpack results safely
                if len(front_result) == 3:
                    front_point, front_conf, front_frame = front_result
                else:
                    print(f"Warning: front_result has {len(front_result)} values instead of 3")
                    front_point, front_conf, front_frame = front_result[0], front_result[1], front_result[2]

                if len(side_result) == 3:
                    side_point, side_conf, side_frame = side_result
                else:
                    print(f"Warning: side_result has {len(side_result)} values instead of 3")
                    side_point, side_conf, side_frame = side_result[0], side_result[1], side_result[2]

                if len(floor_result) == 3:
                    floor_point, floor_conf, floor_frame = floor_result
                else:
                    print(f"Warning: floor_result has {len(floor_result)} values instead of 3")
                    floor_point, floor_conf, floor_frame = floor_result[0], floor_result[1], floor_result[2]
            except Exception as e:
                print(f"Error detecting objects: {e}")
                print(f"Front result: {front_result}")
                print(f"Side result: {side_result}")
                print(f"Floor result: {floor_result}")
                return False

            # Store confidence scores
            self.camera_confidences.append((front_conf, side_conf, floor_conf))

            # Handle camera calibration if active
            if self.calibration_active:
                try:
                    calibration_complete = self.calibrate_cameras(front_point, side_point)
                    if calibration_complete:
                        print("[SYSTEM] Y-coordinate calibration complete. Starting normal tracking.")
                except Exception as e:
                    print(f"Error during calibration: {e}")
                    self.calibration_active = False
                    self.calibration_complete = True
                    print("[SYSTEM] Calibration error. Switching to normal tracking with default settings.")
            else:
                # Verify Y-coordinate directions during normal operation
                if front_point is not None and side_point is not None:
                    try:
                        front_y, side_y, agreement, correlation = self.verify_y_coordinate_directions(
                            front_point,
                            side_point)
                    except Exception as e:
                        print(f"Error verifying Y-coordinates: {e}")

            # Resize frames for display
            try:
                disp_width = self.config['camera_width']
                disp_height = self.config['camera_height']
                front_resized = cv2.resize(front_frame, (disp_width, disp_height))
                side_resized = cv2.resize(side_frame, (disp_width, disp_height))
                floor_resized = cv2.resize(floor_frame, (disp_width, disp_height))
            except Exception as e:
                print(f"Error resizing frames: {e}")
                return False

            # Add Y-verification info to frames if the camera debug info is enabled
            if self.config['camera_debug_info']:
                try:
                    # Add calibration status if in calibration mode
                    if self.calibration_active:
                        # Calibration progress bar
                        max_frames = self.config['y_calibration_frames']
                        progress = min(100, int(self.calibration_frame_count / max_frames * 100))

                        bar_width = int(disp_width * 0.8)
                        filled_width = int(bar_width * progress / 100)

                        cv2.rectangle(front_resized, (50, disp_height - 50), (50 + bar_width, disp_height - 40),
                                      (0, 0, 0),
                                      1)
                        cv2.rectangle(front_resized, (50, disp_height - 50),
                                      (50 + filled_width, disp_height - 40),
                                      (0, 255, 0), -1)

                        cv2.rectangle(side_resized, (50, disp_height - 50), (50 + bar_width, disp_height - 40),
                                      (0, 0, 0),
                                      1)
                        cv2.rectangle(side_resized, (50, disp_height - 50),
                                      (50 + filled_width, disp_height - 40),
                                      (0, 255, 0), -1)

                        # Add calibration text
                        cv2.putText(front_resized, f"CALIBRATING: {progress}%", (50, disp_height - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(side_resized, f"CALIBRATING: {progress}%", (50, disp_height - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Instructions
                        cv2.putText(front_resized, "Move object up and down", (50, disp_height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(side_resized, "Move object up and down", (50, disp_height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Y direction indicators for normal operation
                    elif self.calibration_complete:
                        # Draw Y direction arrow on front camera
                        arrow_start = (disp_width - 60, disp_height - 40)
                        if self.front_y_direction > 0:
                            arrow_end = (disp_width - 60, disp_height - 80)
                        else:
                            arrow_end = (disp_width - 60, disp_height)
                        cv2.arrowedLine(front_resized, arrow_start, arrow_end, (0, 255, 0), 3, tipLength=0.3)
                        cv2.putText(front_resized, f"Y: {self.front_y_direction}",
                                    (disp_width - 100, disp_height - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Draw Y direction arrow on side camera (X in side view)
                        arrow_start = (disp_width - 60, disp_height - 40)
                        if self.side_y_direction > 0:
                            arrow_end = (disp_width - 100, disp_height - 40)
                        else:
                            arrow_end = (disp_width - 20, disp_height - 40)
                        cv2.arrowedLine(side_resized, arrow_start, arrow_end, (255, 0, 255), 3, tipLength=0.3)
                        cv2.putText(side_resized, f"Y: {self.side_y_direction}",
                                    (disp_width - 100, disp_height - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        # Add verification status
                        status_text = self.y_verification_state
                        if len(status_text) > 25:  # Truncate if too long
                            status_text = status_text[:25] + "..."

                        status_color = (0, 255, 0) if "agree" in self.y_verification_state.lower() else (
                            0, 0, 255)
                        cv2.putText(front_resized, status_text, (10, disp_height - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                        cv2.putText(side_resized, status_text, (10, disp_height - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                        # Add correlation info if available
                        if self.y_correlation_scores and len(self.y_correlation_scores) > 0:
                            corr = self.y_correlation_scores[-1]
                            corr_color = (0, 200, 0) if corr > 0.5 else (0, 0, 200)
                            cv2.putText(front_resized, f"Corr: {corr:.2f}", (10, disp_height - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, corr_color, 2)
                            cv2.putText(side_resized, f"Corr: {corr:.2f}", (10, disp_height - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, corr_color, 2)
                except Exception as e:
                    print(f"Error adding camera debug info: {e}")

            # Store raw camera data
            self.raw_camera_data.append({
                'frame': self.frame_count,
                'time': self.timestamps[-1],
                'front': front_point,
                'side': side_point,
                'floor': floor_point,
                'confidences': (front_conf, side_conf, floor_conf)
            })

            # Create Y-coordinate comparison visualization if enabled
            try:
                if self.config['show_comparison_window'] and not self.calibration_active:
                    comparison_vis = self.create_y_comparison_visualization(front_point, side_point,
                                                                            front_frame,
                                                                            side_frame)
                    cv2.imshow('Y-Coordinate Comparison', comparison_vis)
            except Exception as e:
                print(f"Error creating Y-coordinate comparison: {e}")

            # Display the processed frames
            try:
                cv2.imshow('Front Camera (Y)', front_resized)
                cv2.imshow('Side Camera (Y)', side_resized)
                cv2.imshow('Floor Camera (X,Z)', floor_resized)
            except Exception as e:
                print(f"Error displaying frames: {e}")

            # Skip 3D reconstruction during calibration
            if self.calibration_active:
                self.frame_count += 1
                return True

            # Reconstruct 3D position with exclusive camera assignments
            try:
                confidences = (front_conf, side_conf, floor_conf)
                point_3d = self.reconstruct_3d_point(front_point, side_point, floor_point, confidences)

                if point_3d:
                    # Make sure point_3d has exactly 3 values (x, y, z)
                    if len(point_3d) != 3:
                        print(f"Warning: point_3d has {len(point_3d)} values instead of 3. Fixing.")
                        # Extract only the first 3 values if there are more
                        point_3d = point_3d[:3]

                    self.trajectory_3d.append(point_3d)

                    # Calculate dimension movement
                    try:
                        x_movement, y_movement, z_movement = self.calculate_dimension_movement()
                        self.dimension_movements['X'].append(x_movement)
                        self.dimension_movements['Y'].append(y_movement)
                        self.dimension_movements['Z'].append(z_movement)
                    except Exception as e:
                        print(f"Error calculating dimension movement: {e}")

                    # Update visualization
                    try:
                        self.update_visualization()
                    except Exception as e:
                        print(f"Error updating visualization: {e}")

                    # Update dimension limits
                    try:
                        x, y, z = point_3d
                        self.dimension_limits['X'][0] = min(self.dimension_limits['X'][0], x)
                        self.dimension_limits['X'][1] = max(self.dimension_limits['X'][1], x)
                        self.dimension_limits['Y'][0] = min(self.dimension_limits['Y'][0], y)
                        self.dimension_limits['Y'][1] = max(self.dimension_limits['Y'][1], y)
                        self.dimension_limits['Z'][0] = min(self.dimension_limits['Z'][0], z)
                        self.dimension_limits['Z'][1] = max(self.dimension_limits['Z'][1], z)
                    except Exception as e:
                        print(f"Error updating dimension limits: {e}")
            except Exception as e:
                print(f"Error in 3D point reconstruction: {e}")

            # Increment frame counter
            self.frame_count += 1

            # Control frame rate for smoother visualization
            time.sleep(1.0 / 30.0)

            return True

        except Exception as e:
            print(f"Unhandled error in process_frame: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_dimension_movement(self):
        """Calculate the amount of movement in each dimension"""
        if len(self.trajectory_3d) < 2:
            return 0, 0, 0

        # Calculate the movement in each dimension for the last few points
        window_size = min(20, len(self.trajectory_3d))
        recent_points = self.trajectory_3d[-window_size:]

        x_values = [p[0] for p in recent_points]
        y_values = [p[1] for p in recent_points]
        z_values = [p[2] for p in recent_points]

        # Calculate total variation in each dimension
        x_movement = max(x_values) - min(x_values)
        y_movement = max(y_values) - min(y_values)
        z_movement = max(z_values) - min(z_values)

        return x_movement, y_movement, z_movement

    def detect_primary_plane(self):
        """Detect which plane has the most movement"""
        if len(self.dimension_movements['X']) < 10:
            return "Not enough data"

        # Calculate average movement in each dimension
        avg_x = sum(self.dimension_movements['X']) / len(self.dimension_movements['X'])
        avg_y = sum(self.dimension_movements['Y']) / len(self.dimension_movements['Y'])
        avg_z = sum(self.dimension_movements['Z']) / len(self.dimension_movements['Z'])

        # Sort dimensions by movement amount
        movements = [(avg_x, 'X'), (avg_y, 'Y'), (avg_z, 'Z')]
        movements.sort(reverse=True)

        # The dimension with the least movement is perpendicular to the primary plane
        least_movement_dim = movements[2][1]

        # Determine the primary plane
        if least_movement_dim == 'X':
            return "YZ plane"
        elif least_movement_dim == 'Y':
            return "XZ plane"
        else:  # Z
            return "XY plane"

    def analyze_trajectory(self):
        """Analyze the tracked trajectory and return statistics"""
        if not self.trajectory_3d or len(self.trajectory_3d) < 2:
            print("Not enough trajectory data for analysis")
            return {}

        try:
            # Calculate statistics
            stats = {}

            # Movement range in each dimension
            for dim, (min_val, max_val) in self.dimension_limits.items():
                if min_val != float('inf') and max_val != float('-inf'):
                    stats[f'{dim}_range'] = max_val - min_val

            # Calculate average movement in each dimension
            if self.dimension_movements['X']:
                stats['avg_x_movement'] = sum(self.dimension_movements['X']) / len(
                    self.dimension_movements['X'])
                stats['avg_y_movement'] = sum(self.dimension_movements['Y']) / len(
                    self.dimension_movements['Y'])
                stats['avg_z_movement'] = sum(self.dimension_movements['Z']) / len(
                    self.dimension_movements['Z'])

                # Determine the primary plane of movement
                movements = [(stats['avg_x_movement'], 'X'),
                             (stats['avg_y_movement'], 'Y'),
                             (stats['avg_z_movement'], 'Z')]
                movements.sort(reverse=True)

                # The two dimensions with the most movement define the primary plane
                primary_dims = movements[0][1] + movements[1][1]
                if primary_dims in ['XY', 'YX']:
                    stats['primary_plane'] = 'XY'
                elif primary_dims in ['XZ', 'ZX']:
                    stats['primary_plane'] = 'XZ'
                elif primary_dims in ['YZ', 'ZY']:
                    stats['primary_plane'] = 'YZ'

                # Calculate the "2D-ness" of the movement
                # Ratio of least movement to most movement - lower means more 2D-like
                stats['dimensionality_ratio'] = movements[2][0] / movements[0][0]
                stats['is_mostly_2d'] = stats[
                                            'dimensionality_ratio'] < 0.2  # If < 20% movement in perpendicular axis

            # Total distance traveled
            total_distance = 0
            for i in range(1, len(self.trajectory_3d)):
                p1 = self.trajectory_3d[i - 1]
                p2 = self.trajectory_3d[i]
                distance = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
                total_distance += distance

            stats['total_distance'] = total_distance

            # Displacement (straight-line distance from start to end)
            start = self.trajectory_3d[0]
            end = self.trajectory_3d[-1]
            displacement = np.sqrt(
                (end[0] - start[0]) ** 2 +
                (end[1] - start[1]) ** 2 +
                (end[2] - start[2]) ** 2
            )
            stats['displacement'] = displacement

            # Path efficiency (displacement / total_distance)
            if total_distance > 0:
                stats['path_efficiency'] = displacement / total_distance
            else:
                stats['path_efficiency'] = 0

            # Calculate average camera confidences
            if self.camera_confidences:
                avg_front_conf = sum(c[0] for c in self.camera_confidences) / len(self.camera_confidences)
                avg_side_conf = sum(c[1] for c in self.camera_confidences) / len(self.camera_confidences)
                avg_floor_conf = sum(c[2] for c in self.camera_confidences) / len(self.camera_confidences)

                stats['avg_front_confidence'] = avg_front_conf
                stats['avg_side_confidence'] = avg_side_conf
                stats['avg_floor_confidence'] = avg_floor_conf

            # Y-coordinate verification statistics
            if self.y_agreement_scores:
                stats['avg_y_agreement'] = sum(self.y_agreement_scores) / len(self.y_agreement_scores)
                stats['min_y_agreement'] = min(self.y_agreement_scores)
                stats['max_y_agreement'] = max(self.y_agreement_scores)
                stats['final_front_y_direction'] = self.front_y_direction
                stats['final_side_y_direction'] = self.side_y_direction
                stats['y_verification_state'] = self.y_verification_state

                if self.y_correlation_scores:
                    stats['avg_y_correlation'] = sum(self.y_correlation_scores) / len(
                        self.y_correlation_scores)

                if self.y_conflict_frames:
                    stats['y_conflict_count'] = len(self.y_conflict_frames)

            # Add calibration method if available
            if hasattr(self, 'calibration_results') and self.calibration_results:
                if 'method' in self.calibration_results:
                    stats['y_calibration_method'] = self.calibration_results['method']
                if 'y_dominance' in self.calibration_results:
                    stats['y_dominance'] = self.calibration_results['y_dominance']

            # Print statistics
            print("\nTrajectory Analysis:")
            print(f"Total points tracked: {len(self.trajectory_3d)}")

            print("\nDimensional Analysis:")
            for dim in ['X', 'Y', 'Z']:
                dim_range = stats.get(f'{dim}_range', 0)
                print(f"{dim} range: {dim_range:.2f} units")

            if 'primary_plane' in stats:
                print(f"Primary plane of movement: {stats['primary_plane']}")
                print(f"Dimensionality ratio: {stats['dimensionality_ratio']:.4f}")
                print(f"Movement is mostly 2D: {stats['is_mostly_2d']}")

            print(f"\nTotal distance traveled: {stats['total_distance']:.2f} units")
            print(f"Displacement (start to end): {stats['displacement']:.2f} units")
            print(f"Path efficiency: {stats['path_efficiency']:.2f}")

            if 'avg_front_confidence' in stats:
                print("\nAverage Camera Confidence:")
                print(f"Front: {stats['avg_front_confidence']:.2f}")
                print(f"Side: {stats['avg_side_confidence']:.2f}")
                print(f"Floor: {stats['avg_floor_confidence']:.2f}")

            if 'avg_y_agreement' in stats:
                print("\nY-Coordinate Verification:")
                print(f"Average agreement: {stats['avg_y_agreement']:.2f}")
                print(f"Final front Y direction: {stats['final_front_y_direction']}")
                print(f"Final side Y direction: {stats['final_side_y_direction']}")
                print(f"Final verification state: {stats['y_verification_state']}")

                if 'avg_y_correlation' in stats:
                    print(f"Average correlation: {stats['avg_y_correlation']:.2f}")

                if 'y_conflict_count' in stats:
                    print(f"Y conflicts detected: {stats['y_conflict_count']}")

                if 'y_calibration_method' in stats:
                    print(f"Calibration method: {stats['y_calibration_method']}")

            return stats

        except Exception as e:
            print(f"Error in trajectory analysis: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def analyze_trajectory_bounds(self):
        """
        Analyze the trajectory to find its bounds and key points.
        This helps with precise triangle placement.

        Returns:
            dict: Information about trajectory bounds and key points
        """
        if not self.trajectory_3d:
            return {"error": "No trajectory data available"}

        try:
            # Convert to numpy array
            points = np.array(self.trajectory_3d)

            # Find bounding box
            min_bounds = np.min(points, axis=0)
            max_bounds = np.max(points, axis=0)
            center = np.mean(points, axis=0)

            # Find key points
            start_point = points[0]
            end_point = points[-1]

            # Find the point with maximum distance from center
            dists_from_center = np.linalg.norm(points - center, axis=1)
            max_dist_idx = np.argmax(dists_from_center)
            max_dist_point = points[max_dist_idx]

            # Find trajectory quartile points (25%, 50%, 75%)
            point_count = len(points)
            quartile_indices = [
                0,  # Start
                point_count // 4,  # 25%
                point_count // 2,  # 50%
                3 * point_count // 4,  # 75%
                point_count - 1  # End
            ]
            quartile_points = points[quartile_indices]

            # Results
            info = {
                "min_bounds": min_bounds.tolist(),
                "max_bounds": max_bounds.tolist(),
                "center": center.tolist(),
                "width": (max_bounds[0] - min_bounds[0]),
                "height": (max_bounds[1] - min_bounds[1]),
                "depth": (max_bounds[2] - min_bounds[2]),
                "start_point": start_point.tolist(),
                "end_point": end_point.tolist(),
                "max_distance_point": max_dist_point.tolist(),
                "quarter_points": [p.tolist() for p in quartile_points],
                "point_count": point_count
            }

            # Calculate principal components to find main plane orientation
            centered_points = points - center
            cov_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort eigenvectors by eigenvalues (descending)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Add principal directions to results
            info["principal_directions"] = eigenvectors.tolist()

            # Project points onto principal axes to find extents
            projected = np.dot(centered_points, eigenvectors)
            proj_min = np.min(projected, axis=0).tolist()
            proj_max = np.max(projected, axis=0).tolist()

            info["principal_min"] = proj_min
            info["principal_max"] = proj_max

            return info

        except Exception as e:
            print(f"Error analyzing trajectory bounds: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def print_trajectory_info(self):
        """
        Print detailed information about the trajectory to help with triangle placement.
        This provides data needed to position the triangle correctly.
        """
        info = self.analyze_trajectory_bounds()

        if "error" in info:
            print(f"Could not analyze trajectory: {info['error']}")
            return

        print("\n==========================================")
        print("  TRAJECTORY INFORMATION FOR TRIANGLE PLACEMENT")
        print("==========================================")

        print(f"\nTotal points: {info['point_count']}")

        print("\nBounding Box:")
        print(f"  X: {info['min_bounds'][0]:.2f} to {info['max_bounds'][0]:.2f} (width: {info['width']:.2f})")
        print(f"  Y: {info['min_bounds'][1]:.2f} to {info['max_bounds'][1]:.2f} (height: {info['height']:.2f})")
        print(f"  Z: {info['min_bounds'][2]:.2f} to {info['max_bounds'][2]:.2f} (depth: {info['depth']:.2f})")

        print("\nCenter point:")
        print(f"  [{info['center'][0]:.2f}, {info['center'][1]:.2f}, {info['center'][2]:.2f}]")

        print("\nKey Points for Triangle Placement:")
        print("  Start point:")
        print(f"    [{info['start_point'][0]:.2f}, {info['start_point'][1]:.2f}, {info['start_point'][2]:.2f}]")

        print("  End point:")
        print(f"    [{info['end_point'][0]:.2f}, {info['end_point'][1]:.2f}, {info['end_point'][2]:.2f}]")

        print("  25% point:")
        print(
            f"    [{info['quarter_points'][1][0]:.2f}, {info['quarter_points'][1][1]:.2f}, {info['quarter_points'][1][2]:.2f}]")

        print("  50% point (middle):")
        print(
            f"    [{info['quarter_points'][2][0]:.2f}, {info['quarter_points'][2][1]:.2f}, {info['quarter_points'][2][2]:.2f}]")

        print("  75% point:")
        print(
            f"    [{info['quarter_points'][3][0]:.2f}, {info['quarter_points'][3][1]:.2f}, {info['quarter_points'][3][2]:.2f}]")

        print("  Maximum distance point:")
        print(
            f"    [{info['max_distance_point'][0]:.2f}, {info['max_distance_point'][1]:.2f}, {info['max_distance_point'][2]:.2f}]")

        print("\nSuggested Triangle Vertices:")
        # Option 1: Using start, middle, end points
        print("  Option 1 (Start-Middle-End):")
        print(f"  'vertices': [")
        print(f"      {info['start_point']},  # Start point")
        print(f"      {info['quarter_points'][2]},  # Middle point")
        print(f"      {info['end_point']}   # End point")
        print(f"  ]")

        # Option 2: Using start, max distance, end points
        print("\n  Option 2 (Start-MaxDist-End):")
        print(f"  'vertices': [")
        print(f"      {info['start_point']},  # Start point")
        print(f"      {info['max_distance_point']},  # Max distance point")
        print(f"      {info['end_point']}   # End point")
        print(f"  ]")

        # Option 3: Using 25%, center, 75% points
        print("\n  Option 3 (25%-Center-75%):")
        print(f"  'vertices': [")
        print(f"      {info['quarter_points'][1]},  # 25% point")
        print(f"      {info['center']},  # Center")
        print(f"      {info['quarter_points'][3]}   # 75% point")
        print(f"  ]")

        print("\nCopy one of these options into your triangle_params dictionary.")
        print("You can also create your own vertices by selecting any three points along the trajectory.")
        print("==========================================")

        return info

    def create_triangle_presets(self):
        """
        Create a set of triangle presets based on trajectory analysis.

        Returns:
            list: List of triangle parameter dictionaries for different preset options
        """
        info = self.analyze_trajectory_bounds()

        if "error" in info:
            return [{"error": info["error"]}]

        presets = []

        # Preset 1: Start-Middle-End triangle
        presets.append({
            'name': 'start_middle_end',
            'color': 'magenta',
            'transparency': 0.4,
            'vertices': [
                info['start_point'],
                info['quarter_points'][2],  # Middle point (50%)
                info['end_point']
            ]
        })

        # Preset 2: Start-MaxDistance-End triangle
        presets.append({
            'name': 'start_maxdist_end',
            'color': 'cyan',
            'transparency': 0.4,
            'vertices': [
                info['start_point'],
                info['max_distance_point'],
                info['end_point']
            ]
        })

        # Preset 3: 25%-Center-75% triangle
        presets.append({
            'name': 'quartile_triangle',
            'color': 'yellow',
            'transparency': 0.4,
            'vertices': [
                info['quarter_points'][1],  # 25% point
                info['center'],
                info['quarter_points'][3]  # 75% point
            ]
        })

        # Preset 4: Bounding triangle in principal plane
        try:
            # Create a triangle that approximately bounds the projection in the principal plane
            center = np.array(info['center'])
            principal_dirs = np.array(info['principal_directions'])

            # Get the first two principal directions
            u = principal_dirs[:, 0]  # Major axis
            v = principal_dirs[:, 1]  # Minor axis

            # Get extents along principal directions
            u_min, u_max = info['principal_min'][0], info['principal_max'][0]
            v_min, v_max = info['principal_min'][1], info['principal_max'][1]

            # Create a triangle using these extents
            vertices = [
                center + u_min * u + v_min * v,  # Bottom left corner
                center + u_max * u + v_min * v,  # Bottom right corner
                center + (u_max + u_min) / 2 * u + v_max * v  # Top middle
            ]

            presets.append({
                'name': 'bounding_triangle',
                'color': 'green',
                'transparency': 0.4,
                'vertices': vertices.tolist()
            })
        except Exception as e:
            print(f"Could not create bounding triangle preset: {e}")

        return presets
    def save_trajectory(self, base_filename=None):
        """
        Save the tracked trajectory data and analysis

        Args:
            base_filename: Base name for saved files (without extension)

        Returns:
            dict: Dictionary of saved file paths
        """
        if not self.trajectory_3d:
            print("No trajectory data to save")
            return {}

        try:
            # Generate base filename if not provided
            if base_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"exclusive_camera_track_{timestamp}"
            else:
                # Strip extension if provided
                base_filename = os.path.splitext(base_filename)[0]

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(base_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save trajectory data as CSV
            csv_file = f"{base_filename}.csv"
            with open(csv_file, 'w') as f:
                # Write header
                f.write("frame,time,X,Y,Z\n")

                # Write data
                start_time = self.timestamps[0] if self.timestamps else 0
                for i, point in enumerate(self.trajectory_3d):
                    time_val = 0
                    if i < len(self.timestamps):
                        time_val = self.timestamps[i] - start_time

                    f.write(f"{i},{time_val:.4f},{point[0]:.4f},{point[1]:.4f},{point[2]:.4f}\n")

            print(f"3D trajectory saved to {csv_file}")

            # Save analysis data as JSON with custom encoder for NumPy types
            analysis = self.analyze_trajectory()
            json_file = f"{base_filename}_analysis.json"
            with open(json_file, 'w') as f:
                json.dump(analysis, f, indent=4, cls=NumpyEncoder)

            print(f"Analysis saved to {json_file}")

            # Save visualization
            viz_file = f"{base_filename}_viz.png"
            self.save_visualization_image(viz_file)

            # Save Y-coordinate analysis if available
            y_file = f"{base_filename}_y_analysis.json"
            y_data = {
                "front_y_direction": self.front_y_direction,
                "side_y_direction": self.side_y_direction,
                "verification_state": self.y_verification_state,
                "calibration_method": self.calibration_results.get("method", "standard") if hasattr(
                    self,
                    "calibration_results") else "unknown",
                "agreement_scores": self.y_agreement_scores,
                "correlation_scores": self.y_correlation_scores,
                "conflict_frames": self.y_conflict_frames
            }

            with open(y_file, 'w') as f:
                json.dump(y_data, f, indent=4, cls=NumpyEncoder)

            print(f"Y-coordinate analysis saved to {y_file}")

            return {
                'csv': csv_file,
                'analysis': json_file,
                'viz': viz_file,
                'y_analysis': y_file
            }

        except Exception as e:
            print(f"Error saving trajectory: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def export_camera_calibration(self, filename=None):
        """
        Export the camera calibration settings to a JSON file

        Args:
            filename: Name for the output file (without extension)

        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_calibration_{timestamp}"
        else:
            # Strip extension if provided
            filename = os.path.splitext(filename)[0]

        try:
            # Create a dictionary with calibration data
            calibration_data = {
                "front_camera": {
                    "y_direction": self.front_y_direction,
                    "flip_x": self.config['camera_flip']['front_x'],
                    "flip_y": self.config['camera_flip']['front_y']
                },
                "side_camera": {
                    "y_direction": self.side_y_direction,
                    "flip_x": self.config['camera_flip']['side_x'],
                    "flip_y": self.config['camera_flip']['side_y']
                },
                "floor_camera": {
                    "flip_x": self.config['camera_flip']['floor_x'],
                    "flip_y": self.config['camera_flip']['floor_y']
                },
                "calibration_results": self.calibration_results if hasattr(self,
                                                                           'calibration_results') else {},
                "calibration_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera_assignments": {
                    "X": "floor",
                    "Y": "front/side",
                    "Z": "floor"
                }
            }

            # Save to file
            output_file = f"{filename}.json"
            with open(output_file, 'w') as f:
                json.dump(calibration_data, f, indent=4)

            print(f"Camera calibration exported to {output_file}")
            return output_file

        except Exception as e:
            print(f"Error exporting camera calibration: {e}")
            return None

    def save_visualization_image(self, output_file):
        """
        Save a high-quality visualization image of the tracked trajectory

        Args:
            output_file: Path to save the image

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.trajectory_3d:
            print("No trajectory data to visualize")
            return False

        try:
            # Create a new figure for the visualization
            fig = plt.figure(figsize=(15, 12))

            # Extract data
            x_points = [p[0] for p in self.trajectory_3d]
            y_points = [p[1] for p in self.trajectory_3d]
            z_points = [p[2] for p in self.trajectory_3d]

            # Create colormap for time progression
            norm = plt.Normalize(0, len(x_points))
            colors = plt.cm.viridis(norm(range(len(x_points))))

            # 3D Trajectory with enhanced styling
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.set_facecolor('#f5f5f5')  # Light background

            # Plot trajectory with gradient color
            for i in range(1, len(x_points)):
                ax1.plot3D(x_points[i - 1:i + 1], y_points[i - 1:i + 1], z_points[i - 1:i + 1],
                           color=colors[i], linewidth=2)

            # Mark start and end points
            ax1.scatter(x_points[0], y_points[0], z_points[0], color='green', s=100, label='Start',
                        edgecolor='black')
            ax1.scatter(x_points[-1], y_points[-1], z_points[-1], color='red', s=100, label='End',
                        edgecolor='black')

            # Add axis labels
            ax1.set_xlabel('X (Floor Camera)', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Y (Front/Side Cameras)', fontsize=10, fontweight='bold')
            ax1.set_zlabel('Z (Floor Camera)', fontsize=10, fontweight='bold')

            # Add a title with camera assignment info
            primary_plane = self.detect_primary_plane()
            title = f'3D Trajectory - Exclusive Camera Assignment\n'
            title += f'Primary plane: {primary_plane}'

            # Add Y-verification info
            if hasattr(self, 'y_verification_state') and self.y_verification_state:
                if len(self.y_verification_state) > 30:
                    y_status = self.y_verification_state[:30] + "..."
                else:
                    y_status = self.y_verification_state
                title += f'\nY-Status: {y_status}'

            ax1.set_title(title, fontweight='bold')
            ax1.legend()

            # Create three 2D views with enhanced styling
            # XY Plane (Front View)
            ax2 = fig.add_subplot(222)
            ax2.set_facecolor('#f8f8f8')
            ax2.scatter(x_points, y_points, c=colors, s=30, alpha=0.7)
            ax2.plot(x_points, y_points, 'k-', alpha=0.3, linewidth=1)
            ax2.scatter(x_points[0], y_points[0], color='green', s=100, marker='o', label='Start',
                        edgecolor='black')
            ax2.scatter(x_points[-1], y_points[-1], color='red', s=100, marker='o', label='End',
                        edgecolor='black')
            ax2.set_xlabel('X (Floor)', fontweight='bold')
            ax2.set_ylabel('Y (Front/Side)', fontweight='bold')
            ax2.set_title('Front View (X-Y)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # XZ Plane (Floor View)
            ax3 = fig.add_subplot(223)
            ax3.set_facecolor('#f8f8f8')
            ax3.scatter(x_points, z_points, c=colors, s=30, alpha=0.7)
            ax3.plot(x_points, z_points, 'k-', alpha=0.3, linewidth=1)
            ax3.scatter(x_points[0], z_points[0], color='green', s=100, marker='o', label='Start',
                        edgecolor='black')
            ax3.scatter(x_points[-1], z_points[-1], color='red', s=100, marker='o', label='End',
                        edgecolor='black')
            ax3.set_xlabel('X (Floor)', fontweight='bold')
            ax3.set_ylabel('Z (Floor)', fontweight='bold')
            ax3.set_title('Floor View (X-Z)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # Y Verification Plot
            ax4 = fig.add_subplot(224)
            ax4.set_facecolor('#f8f8f8')

            if self.front_y_values and self.side_y_values:
                # Get appropriate window size for display
                window_size = min(100, len(self.front_y_values))
                display_start = max(0, len(self.front_y_values) - window_size)

                # Extract display window
                frames_y = list(range(display_start, display_start + window_size))
                display_front_y = self.front_y_values[display_start:display_start + window_size]
                display_side_y = self.side_y_values[display_start:display_start + window_size]

                # Plot Y values
                ax4.plot(frames_y, display_front_y, 'g-',
                         label=f'Front Y (Dir: {self.front_y_direction})', linewidth=2)
                ax4.plot(frames_y, display_side_y, 'm-',
                         label=f'Side Y (Dir: {self.side_y_direction})', linewidth=2)

                # Plot agreement scores if available
                if self.y_agreement_scores and len(self.y_agreement_scores) >= display_start:
                    display_agreement = self.y_agreement_scores[
                                        display_start:display_start + window_size]
                    display_frames = frames_y[:len(display_agreement)]
                    ax4.plot(display_frames, display_agreement, 'k--', label='Agreement', alpha=0.7)

                # Mark conflict frames
                if self.y_conflict_frames:
                    conflict_frames = [f for f in self.y_conflict_frames if
                                       f >= display_start and f < display_start + window_size]
                    if conflict_frames:
                        for cf in conflict_frames:
                            ax4.axvline(x=cf, color='r', linestyle='--', alpha=0.3)

                ax4.set_xlabel('Frame', fontweight='bold')
                ax4.set_ylabel('Y Value / Agreement', fontweight='bold')
                ax4.set_title('Y-Coordinate Verification', fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, "No Y verification data available",
                         ha='center', va='center', transform=ax4.transAxes,
                         fontsize=12)
                ax4.set_title('Y-Coordinate Verification', fontweight='bold')

            # Add colorbar for time progression
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3, ax4], orientation='horizontal',
                                pad=0.05, aspect=40, shrink=0.6)

            cbar.set_label('Frame Progression', fontweight='bold')

            # Add tracking stats
            fig.text(0.5, 0.01,
                     f"Total Frames: {self.frame_count}   Points Tracked: {len(self.trajectory_3d)}   "
                     f"Front Y-Dir: {self.front_y_direction}   Side Y-Dir: {self.side_y_direction}",
                     ha='center', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

            plt.tight_layout()

            # Save high-resolution image
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Visualization saved to {output_file}")
            return True

        except Exception as e:
            print(f"Error saving visualization: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_report(self, output_file=None):
        """
        Generate a detailed HTML report of the tracking session

        Args:
            output_file: Path for the output HTML file

        Returns:
            str: Path to the generated report file
        """
        if not self.trajectory_3d:
            print("No trajectory data available for report generation")
            return None

        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"tracking_report_{timestamp}.html"

            # Analyze trajectory
            stats = self.analyze_trajectory()

            # Save visualization image
            img_path = f"{os.path.splitext(output_file)[0]}_viz.png"
            self.save_visualization_image(img_path)

            # Create HTML content
            html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>3D Camera Tracking Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #3498db; }}
                        .section {{ margin-bottom: 30px; }}
                        .stats {{ display: flex; flex-wrap: wrap; }}
                        .stat-item {{ width: 200px; margin: 10px; padding: 10px; 
                                   background-color: #f8f9fa; border-radius: 5px; }}
                        .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                        .stat-label {{ color: #7f8c8d; }}
                        img {{ max-width: 100%; border: 1px solid #ddd; }}
                        .camera-info {{ display: flex; }}
                        .camera {{ width: 30%; margin: 10px; padding: 10px; background-color: #f8f9fa; }}
                        .calibration {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>3D Camera Tracking Report</h1>
                    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                    <div class="section">
                        <h2>Trajectory Visualization</h2>
                        <img src="{os.path.basename(img_path)}" alt="Trajectory Visualization">
                    </div>

                    <div class="section">
                        <h2>Tracking Statistics</h2>
                        <div class="stats">
                            <div class="stat-item">
                                <div class="stat-value">{len(self.trajectory_3d)}</div>
                                <div class="stat-label">Total Tracked Points</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{stats.get('total_distance', 0):.2f}</div>
                                <div class="stat-label">Total Distance (units)</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{stats.get('displacement', 0):.2f}</div>
                                <div class="stat-label">Displacement (units)</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{stats.get('path_efficiency', 0):.2f}</div>
                                <div class="stat-label">Path Efficiency</div>
                            </div>
                        </div>

                        <h3>Dimensional Analysis</h3>
                        <table>
                            <tr>
                                <th>Dimension</th>
                                <th>Range</th>
                                <th>Average Movement</th>
                            </tr>
                            <tr>
                                <td>X</td>
                                <td>{stats.get('X_range', 0):.2f}</td>
                                <td>{stats.get('avg_x_movement', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>Y</td>
                                <td>{stats.get('Y_range', 0):.2f}</td>
                                <td>{stats.get('avg_y_movement', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>Z</td>
                                <td>{stats.get('Z_range', 0):.2f}</td>
                                <td>{stats.get('avg_z_movement', 0):.2f}</td>
                            </tr>
                        </table>

                        <h3>Primary Plane: {stats.get('primary_plane', 'Unknown')}</h3>
                        <p>Movement is {'' if stats.get('is_mostly_2d', False) else 'not '}mostly 2D
                           (Dimensionality ratio: {stats.get('dimensionality_ratio', 0):.4f})</p>
                    </div>

                    <div class="section">
                        <h2>Camera Configuration</h2>
                        <div class="camera-info">
                            <div class="camera">
                                <h3>Front Camera</h3>
                                <p>Resolution: {self.width_front}x{self.height_front}</p>
                                <p>Controls: Y coordinate</p>
                                <p>Y Direction: {self.front_y_direction}</p>
                                <p>Confidence: {stats.get('avg_front_confidence', 0):.2f}</p>
                            </div>
                            <div class="camera">
                                <h3>Side Camera</h3>
                                <p>Resolution: {self.width_side}x{self.height_side}</p>
                                <p>Controls: Y coordinate</p>
                                <p>Y Direction: {self.side_y_direction}</p>
                                <p>Confidence: {stats.get('avg_side_confidence', 0):.2f}</p>
                            </div>
                            <div class="camera">
                                <h3>Floor Camera</h3>
                                <p>Resolution: {self.width_floor}x{self.height_floor}</p>
                                <p>Controls: X and Z coordinates</p>
                                <p>Confidence: {stats.get('avg_floor_confidence', 0):.2f}</p>
                            </div>
                        </div>
                    </div>

                    <div class="section calibration">
                        <h2>Y-Coordinate Calibration</h2>
                        <p><strong>Verification State:</strong> {self.y_verification_state}</p>
                """

            # Add calibration results if available
            if hasattr(self, 'calibration_results') and self.calibration_results:
                html_content += f"""
                        <h3>Calibration Results</h3>
                        <p><strong>Success:</strong> {self.calibration_results.get('success', False)}</p>
                    """

                if self.calibration_results.get('success', False):
                    html_content += f"""
                            <p><strong>Agreement Percentage:</strong> {self.calibration_results.get('agreement_percentage', 0):.2f}</p>
                            <p><strong>Correlation:</strong> {self.calibration_results.get('correlation', 0):.2f}</p>
                            <p><strong>Movements Analyzed:</strong> {self.calibration_results.get('movements_analyzed', 0)}</p>
                        """
                else:
                    html_content += f"""
                            <p><strong>Reason:</strong> {self.calibration_results.get('reason', 'Unknown')}</p>
                        """

            # Add Y-verification statistics
            if self.y_agreement_scores:
                avg_agreement = sum(self.y_agreement_scores) / len(self.y_agreement_scores)
                html_content += f"""
                        <h3>Y-Verification Statistics</h3>
                        <p><strong>Average Agreement Score:</strong> {avg_agreement:.2f}</p>
                    """

                if self.y_correlation_scores:
                    avg_correlation = sum(self.y_correlation_scores) / len(
                        self.y_correlation_scores)
                    html_content += f"""
                            <p><strong>Average Correlation:</strong> {avg_correlation:.2f}</p>
                        """

                if self.y_conflict_frames:
                    html_content += f"""
                            <p><strong>Conflict Frames:</strong> {len(self.y_conflict_frames)}</p>
                        """

            # Close HTML
            html_content += """
                    </div>
                </body>
                </html>
                """

            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)

            print(f"Tracking report generated at {output_file}")
            return output_file

        except Exception as e:
            print(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return None

    def rotate_trajectory_to_xy_plane(self):
        """
        Rotates the trajectory so that the primary plane of motion becomes parallel to the XY plane.
        This is useful for normalizing trajectories for easier analysis and visualization.

        Returns:
            list: A new list of (x,y,z) points representing the rotated trajectory
            dict: Rotation information including the rotation matrix and primary plane
        """
        if len(self.trajectory_3d) < 3:
            print("Not enough points to determine primary plane and perform rotation")
            return self.trajectory_3d.copy(), {"success": False, "reason": "Not enough points"}

        try:
            # Convert trajectory to numpy array for easier manipulation
            points = np.array(self.trajectory_3d)

            # Step 1: Find the principal components (PCA) to identify the primary plane
            # Center the points
            centered_points = points - np.mean(points, axis=0)

            # Get the covariance matrix
            cov_matrix = np.cov(centered_points, rowvar=False)

            # Get eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort eigenvectors by eigenvalues in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # The eigenvectors with the two largest eigenvalues define the primary plane
            # The eigenvector with the smallest eigenvalue is the normal to this plane
            primary_plane_normal = eigenvectors[:, 2]  # Third column (index 2) is the normal

            # Normalize the normal vector
            primary_plane_normal = primary_plane_normal / np.linalg.norm(primary_plane_normal)

            # Step 2: Calculate rotation matrix to align this normal with the Z-axis (0,0,1)
            z_axis = np.array([0, 0, 1])

            # Calculate rotation axis (cross product) and angle
            rotation_axis = np.cross(primary_plane_normal, z_axis)

            # If rotation_axis is near zero, points are already in XY plane or on XZ/YZ plane
            if np.linalg.norm(rotation_axis) < 1e-10:
                # Check if already aligned with Z or anti-aligned
                if np.dot(primary_plane_normal, z_axis) > 0:
                    print("Trajectory already lies in XY plane")
                    return self.trajectory_3d.copy(), {
                        "success": True,
                        "reason": "Already aligned with XY plane",
                        "primary_plane_normal": primary_plane_normal,
                        "rotation_matrix": np.eye(3)
                    }
                else:
                    # Anti-aligned: rotate 180 degrees around X or Y axis
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
            else:
                # Normalize rotation axis
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                # Calculate rotation angle (dot product)
                cos_angle = np.dot(primary_plane_normal, z_axis)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                # Build the rotation matrix using Rodrigues' rotation formula
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])

                rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

            # Step 3: Apply rotation to all points
            # First adjust to center
            center = np.mean(points, axis=0)
            centered_points = points - center

            # Apply rotation
            rotated_points = np.dot(centered_points, rotation_matrix.T)

            # Restore center
            rotated_points = rotated_points + center

            # Convert back to list of tuples
            rotated_trajectory = [(float(p[0]), float(p[1]), float(p[2])) for p in rotated_points]

            # Calculate the primary plane equation (ax + by + cz + d = 0)
            # Normal is (a,b,c), and d = -dot(normal, point_on_plane)
            d = -np.dot(primary_plane_normal, center)
            plane_equation = (*primary_plane_normal, d)

            # Calculate variance along each axis after rotation
            variances = np.var(rotated_points, axis=0)
            primary_dimensions = np.argsort(variances)[::-1][:2]  # Top 2 dimensions
            primary_dimensions = ['X', 'Y', 'Z'][primary_dimensions[0]] + ['X', 'Y', 'Z'][primary_dimensions[1]]

            # Calculate alignment quality (how well the rotated points lie in the XY plane)
            z_variance = np.var(rotated_points[:, 2])
            xy_variance = np.var(rotated_points[:, 0]) + np.var(rotated_points[:, 1])
            alignment_quality = 1.0 - (z_variance / (z_variance + xy_variance)) if (
                                                                                               z_variance + xy_variance) > 0 else 1.0

            # Store rotation information
            rotation_info = {
                "success": True,
                "center": center.tolist(),
                "primary_plane_normal": primary_plane_normal.tolist(),
                "plane_equation": plane_equation,
                "rotation_matrix": rotation_matrix.tolist(),
                "primary_dimensions_before": self.detect_primary_plane(),
                "primary_dimensions_after": f"XY plane ({primary_dimensions})",
                "alignment_quality": alignment_quality,
                "eigenvalues": eigenvalues.tolist(),
                "original_variance": np.var(centered_points, axis=0).tolist(),
                "rotated_variance": variances.tolist()
            }

            print(f"Trajectory rotated to XY plane (Alignment quality: {alignment_quality:.4f})")
            print(f"Original primary plane: {rotation_info['primary_dimensions_before']}")
            print(f"Rotated to: {rotation_info['primary_dimensions_after']}")

            return rotated_trajectory, rotation_info

        except Exception as e:
            print(f"Error rotating trajectory: {e}")
            import traceback
            traceback.print_exc()
            return self.trajectory_3d.copy(), {"success": False, "reason": str(e)}

    def save_rotated_trajectory(self, rotated_trajectory, rotation_info, base_filename=None):
        """
        Save the rotated trajectory and rotation information

        Args:
            rotated_trajectory: List of (x,y,z) rotated points
            rotation_info: Dictionary containing rotation information
            base_filename: Base name for output files (without extension)

        Returns:
            dict: Dictionary of saved file paths
        """
        if not rotated_trajectory:
            print("No rotated trajectory data to save")
            return {}

        try:
            # Generate base filename if not provided
            if base_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"rotated_trajectory_{timestamp}"
            else:
                # Strip extension if provided
                base_filename = os.path.splitext(base_filename)[0]

            # Add _rotated suffix if not already present
            if not base_filename.endswith("_rotated"):
                base_filename = f"{base_filename}_rotated"

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(base_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save rotated trajectory data as CSV
            csv_file = f"{base_filename}.csv"
            with open(csv_file, 'w') as f:
                # Write header
                f.write("frame,X,Y,Z\n")

                # Write data
                for i, point in enumerate(rotated_trajectory):
                    f.write(f"{i},{point[0]:.4f},{point[1]:.4f},{point[2]:.4f}\n")

            print(f"Rotated 3D trajectory saved to {csv_file}")

            # Save rotation information as JSON
            json_file = f"{base_filename}_info.json"
            with open(json_file, 'w') as f:
                json.dump(rotation_info, f, indent=4, cls=NumpyEncoder)

            print(f"Rotation information saved to {json_file}")

            # Save visualization of rotated trajectory
            viz_file = f"{base_filename}_viz.png"
            self.save_rotated_visualization(rotated_trajectory, rotation_info, viz_file)

            return {
                'csv': csv_file,
                'info': json_file,
                'viz': viz_file
            }

        except Exception as e:
            print(f"Error saving rotated trajectory: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def create_reference_triangle(self, params=None):
        """
        Create a reference triangle that can be adjusted to match desired trajectory.
        The triangle can be used to compare actual vs desired motion patterns.

        Args:
            params: Dictionary with triangle parameters:
                   - position: [x, y, z] center of the triangle
                   - size: float, scale factor for triangle size
                   - orientation: [rx, ry, rz] rotation angles in degrees
                   - color: color for the triangle
                   - fit_to_trajectory: if True, auto-fit triangle to trajectory
                   - shape_match: if True, try to match the trajectory shape
                   - vertices: directly specify triangle vertices (overrides other params)

        Returns:
            dict: Triangle information including vertices
        """
        # Default parameters
        if not params:
            params = {}

        default_params = {
            'position': [0, 0, 0],  # Default center at origin
            'size': 10.0,  # Default size
            'orientation': [0, 0, 0],  # Default orientation
            'color': 'magenta',  # Default color
            'fit_to_trajectory': True,  # Auto-fit to trajectory by default
            'transparency': 0.4,  # Default transparency
            'view_optimization': True,  # Optimize for visibility in all views
            'shape_match': True  # Try to match trajectory shape
        }

        # Update with provided parameters
        for key, value in default_params.items():
            if key not in params:
                params[key] = value

        try:
            # If custom vertices are provided, use them directly
            if 'vertices' in params and params['vertices'] is not None:
                vertices = np.array(params['vertices'])
                position = np.mean(vertices, axis=0)

                # Create edges
                edges = [(0, 1), (1, 2), (2, 0)]

                # Calculate normal vector
                v1 = vertices[1] - vertices[0]
                v2 = vertices[2] - vertices[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)

                # Calculate area
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))

                return {
                    'vertices': vertices.tolist(),
                    'edges': edges,
                    'normal': normal.tolist(),
                    'position': position.tolist(),
                    'size': np.linalg.norm(v1),  # Approximate size
                    'color': params['color'],
                    'area': area,
                    'transparency': params['transparency'],
                    'custom': True
                }

            # If trajectory exists and auto-fit is enabled, use trajectory to determine triangle properties
            if self.trajectory_3d and params['fit_to_trajectory']:
                points = np.array(self.trajectory_3d)

                # If we should match the shape of the trajectory
                if params['shape_match']:
                    # Find key points in the trajectory to form a triangle
                    # We'll try different strategies to find meaningful points

                    # Strategy 1: Use start, end, and point of maximum deviation
                    start_point = points[0]
                    end_point = points[-1]

                    # Find point of maximum deviation from the start-end line
                    start_end_vector = end_point - start_point
                    start_end_length = np.linalg.norm(start_end_vector)

                    if start_end_length > 0:
                        # Unit vector along start-end line
                        line_dir = start_end_vector / start_end_length

                        # Calculate distances from each point to the line
                        max_dist = 0
                        max_point_idx = 0

                        for i, point in enumerate(points):
                            # Vector from start to current point
                            point_vector = point - start_point

                            # Projection onto line direction
                            proj_length = np.dot(point_vector, line_dir)

                            # If projection is within line segment
                            if 0 <= proj_length <= start_end_length:
                                # Calculate perpendicular distance
                                proj_point = start_point + line_dir * proj_length
                                dist = np.linalg.norm(point - proj_point)

                                if dist > max_dist:
                                    max_dist = dist
                                    max_point_idx = i

                        # If we found a point with significant deviation, use it
                        if max_dist > 0.05 * start_end_length:
                            mid_point = points[max_point_idx]

                            # Create triangle from these three points
                            vertices = np.array([start_point, mid_point, end_point])

                            # Calculate center as average of vertices
                            position = np.mean(vertices, axis=0)

                            # Create edges
                            edges = [(0, 1), (1, 2), (2, 0)]

                            # Calculate normal vector
                            v1 = vertices[1] - vertices[0]
                            v2 = vertices[2] - vertices[0]
                            normal = np.cross(v1, v2)
                            normal = normal / np.linalg.norm(normal)

                            # Calculate area
                            area = 0.5 * np.linalg.norm(np.cross(v1, v2))

                            return {
                                'vertices': vertices.tolist(),
                                'edges': edges,
                                'normal': normal.tolist(),
                                'position': position.tolist(),
                                'color': params['color'],
                                'area': area,
                                'transparency': params['transparency'],
                                'shape_matched': True
                            }

                    # Strategy 2: Use points at equal intervals from the trajectory
                    # This works better for circular or complex paths
                    num_points = len(points)
                    if num_points >= 3:
                        # Pick 3 points at roughly equal intervals
                        indices = [0, num_points // 3, 2 * num_points // 3]
                        vertices = points[indices]

                        # Check if these points form a non-degenerate triangle
                        v1 = vertices[1] - vertices[0]
                        v2 = vertices[2] - vertices[0]
                        area = 0.5 * np.linalg.norm(np.cross(v1, v2))

                        if area > 0.01:  # Non-degenerate triangle
                            # Calculate center as average of vertices
                            position = np.mean(vertices, axis=0)

                            # Create edges
                            edges = [(0, 1), (1, 2), (2, 0)]

                            # Calculate normal vector
                            normal = np.cross(v1, v2)
                            normal = normal / np.linalg.norm(normal)

                            return {
                                'vertices': vertices.tolist(),
                                'edges': edges,
                                'normal': normal.tolist(),
                                'position': position.tolist(),
                                'color': params['color'],
                                'area': area,
                                'transparency': params['transparency'],
                                'shape_matched': True
                            }

                # If we get here, either shape_match is False or strategies failed
                # Fall back to PCA-based triangle creation
                min_bounds = np.min(points, axis=0)
                max_bounds = np.max(points, axis=0)
                center = (min_bounds + max_bounds) / 2

                # Get dominant plane using PCA
                try:
                    # Center the points
                    centered_points = points - center

                    # Perform PCA to find principal components
                    cov_matrix = np.cov(centered_points, rowvar=False)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                    # Sort eigenvectors by eigenvalues (descending)
                    idx = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]

                    # First two eigenvectors define the dominant plane
                    u = eigenvectors[:, 0]  # Primary direction
                    v = eigenvectors[:, 1]  # Secondary direction

                    # Get extents along principal directions
                    proj_u = np.dot(centered_points, u)
                    proj_v = np.dot(centered_points, v)

                    u_extent = max(proj_u) - min(proj_u)
                    v_extent = max(proj_v) - min(proj_v)

                    # Create a triangle in the dominant plane
                    size_u = u_extent * 0.5  # Scale factor for u direction
                    size_v = v_extent * 0.5  # Scale factor for v direction

                    # Create triangle vertices in the principal plane
                    vertices = np.array([
                        center - size_u * u - size_v * v,  # Bottom left
                        center + size_u * u - size_v * v,  # Bottom right
                        center + size_v * v  # Top center
                    ])

                    # Create edges
                    edges = [(0, 1), (1, 2), (2, 0)]

                    # Calculate normal vector
                    v1 = vertices[1] - vertices[0]
                    v2 = vertices[2] - vertices[0]
                    normal = np.cross(v1, v2)
                    normal = normal / np.linalg.norm(normal)

                    # Calculate area
                    area = 0.5 * np.linalg.norm(np.cross(v1, v2))

                    # Calculate approximate orientation
                    normal = eigenvectors[:, 2]  # Normal to dominant plane
                    rx = np.arctan2(normal[2], normal[1]) * 180 / np.pi
                    ry = np.arctan2(normal[0], normal[2]) * 180 / np.pi
                    rz = np.arctan2(normal[1], normal[0]) * 180 / np.pi

                    return {
                        'vertices': vertices.tolist(),
                        'edges': edges,
                        'normal': normal.tolist(),
                        'position': center.tolist(),
                        'orientation': [rx, ry, rz],
                        'size': (size_u + size_v) / 2,  # Average size
                        'color': params['color'],
                        'area': area,
                        'transparency': params['transparency'],
                        'pca_fit': True
                    }

                except Exception as e:
                    print(f"Warning: PCA-based triangle fitting failed: {e}")
                    # Fall through to standard triangle creation

            # If we get here, use standard triangle creation with manual parameters
            position = np.array(params['position'])
            size = params['size']

            # Create a balanced triangle with some 3D structure for better visibility if requested
            if params['view_optimization']:
                vertices = np.array([
                    [0, -size / 2, -size / 4],  # Bottom with z-offset
                    [-size / 2, size / 2, size / 4],  # Top left with opposite z
                    [size / 2, size / 2, 0]  # Top right
                ])
            else:
                vertices = np.array([
                    [0, -size / 2, 0],  # Bottom center
                    [-size / 2, size / 2, 0],  # Top left
                    [size / 2, size / 2, 0]  # Top right
                ])

            # Apply rotation
            rx, ry, rz = np.array(params['orientation']) * np.pi / 180  # Convert to radians

            # Rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])

            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])

            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ])

            # Combined rotation (order: Rx → Ry → Rz)
            R = np.dot(Rz, np.dot(Ry, Rx))

            # Apply rotation and translation to vertices
            rotated_vertices = np.dot(vertices, R.T) + position

            # Create edges (pairs of vertices that form the triangle edges)
            edges = [
                (0, 1),  # Bottom to top left
                (1, 2),  # Top left to top right
                (2, 0)  # Top right to bottom
            ]

            # Calculate normal vector of the triangle
            v1 = rotated_vertices[1] - rotated_vertices[0]
            v2 = rotated_vertices[2] - rotated_vertices[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            # Calculate area of the triangle
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))

            # Store triangle information
            triangle_info = {
                'vertices': rotated_vertices.tolist(),
                'edges': edges,
                'normal': normal.tolist(),
                'position': position.tolist(),
                'size': size,
                'orientation': params['orientation'],
                'color': params['color'],
                'area': area,
                'transparency': params['transparency'],
                'manual': True
            }

            return triangle_info

        except Exception as e:
            print(f"Error creating reference triangle: {e}")
            import traceback
            traceback.print_exc()

            # Return a default triangle at origin if there was an error
            return {
                'vertices': [
                    [0, -5, 0],
                    [-5, 5, 0],
                    [5, 5, 0]
                ],
                'edges': [(0, 1), (1, 2), (2, 0)],
                'normal': [0, 0, 1],
                'position': [0, 0, 0],
                'size': 10.0,
                'orientation': [0, 0, 0],
                'color': params['color'],
                'error': str(e),
                'transparency': params['transparency']
            }

    def visualize_with_reference_triangle(self, trajectory, triangle_info, rotated=False, output_file=None):
        """
        Visualize the trajectory with a reference triangle for comparison

        Args:
            trajectory: List of (x,y,z) trajectory points
            triangle_info: Dictionary with triangle information
            rotated: True if this is visualizing the rotated trajectory
            output_file: Path to save visualization image

        Returns:
            bool: True if successful, False otherwise
        """
        if not trajectory:
            print("No trajectory data to visualize")
            return False

        try:
            # Verify we have the required imports
            try:
                from matplotlib.patches import Polygon
                from mpl_toolkits.mplot3d import art3d
            except ImportError as e:
                print(f"Missing imports for triangle visualization: {e}")
                print(
                    "Please add 'from matplotlib.patches import Polygon' and 'from mpl_toolkits.mplot3d import art3d' to your imports")
                return False

            # Create figure
            fig = plt.figure(figsize=(12, 10))

            # Extract trajectory data
            x = [p[0] for p in trajectory]
            y = [p[1] for p in trajectory]
            z = [p[2] for p in trajectory]

            # Create colormap for time progression
            norm = plt.Normalize(0, len(x))
            colors = plt.cm.viridis(norm(range(len(x))))

            # Create 3D axis
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.set_facecolor('#f5f5f5')

            # Plot trajectory
            for i in range(1, len(x)):
                ax1.plot3D(x[i - 1:i + 1], y[i - 1:i + 1], z[i - 1:i + 1],
                           color=colors[i], linewidth=2)

            # Mark start and end points
            ax1.scatter(x[0], y[0], z[0], color='green', s=80, label='Start', edgecolor='black')
            ax1.scatter(x[-1], y[-1], z[-1], color='red', s=80, label='End', edgecolor='black')

            # Plot reference triangle
            vertices = np.array(triangle_info['vertices'])
            edges = triangle_info['edges']
            triangle_color = triangle_info['color']
            transparency = triangle_info['transparency']

            # Print vertices for debugging
            print(f"Triangle vertices: {vertices}")

            # Draw triangle edges
            for e in edges:
                ax1.plot3D(
                    [vertices[e[0]][0], vertices[e[1]][0]],
                    [vertices[e[0]][1], vertices[e[1]][1]],
                    [vertices[e[0]][2], vertices[e[1]][2]],
                    color=triangle_color, linewidth=2, alpha=0.7
                )

            # Create a triangular surface
            # Create a Poly3DCollection for the 3D view
            triangle_coords = vertices  # The 3D coordinates
            triangle_3d = art3d.Poly3DCollection([triangle_coords], alpha=transparency, color=triangle_color)
            ax1.add_collection3d(triangle_3d)

            # Set axis labels
            ax1.set_xlabel('X', fontsize=10)
            ax1.set_ylabel('Y', fontsize=10)
            ax1.set_zlabel('Z', fontsize=10)

            # Set title based on whether this is original or rotated
            title = 'Rotated Trajectory with Reference Triangle' if rotated else 'Original Trajectory with Reference Triangle'
            ax1.set_title(title, fontweight='bold')

            # Add legend
            ax1.legend()

            # Create 2D views
            # XY Plane (Top View)
            ax2 = fig.add_subplot(222)
            ax2.set_facecolor('#f8f8f8')
            ax2.scatter(x, y, c=colors, s=30, alpha=0.7)
            ax2.plot(x, y, 'k-', alpha=0.3, linewidth=1)

            # Plot triangle in XY plane
            for e in edges:
                ax2.plot(
                    [vertices[e[0]][0], vertices[e[1]][0]],
                    [vertices[e[0]][1], vertices[e[1]][1]],
                    color=triangle_color, linewidth=2, alpha=0.7
                )

            # Fill triangle area in XY plane with a Polygon
            xy_vertices = vertices[:, :2]  # Extract x,y coordinates only
            print(f"XY vertices: {xy_vertices}")
            try:
                tri_xy = Polygon(xy_vertices, alpha=transparency, color=triangle_color)
                ax2.add_patch(tri_xy)
            except Exception as e:
                print(f"Could not create XY triangle: {e}")

            # Add start and end points
            ax2.scatter(x[0], y[0], color='green', s=80, label='Start', edgecolor='black')
            ax2.scatter(x[-1], y[-1], color='red', s=80, label='End', edgecolor='black')

            ax2.set_xlabel('X', fontweight='bold')
            ax2.set_ylabel('Y', fontweight='bold')
            ax2.set_title('Top View (XY Plane)', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # XZ Plane (Front View)
            ax3 = fig.add_subplot(223)
            ax3.set_facecolor('#f8f8f8')
            ax3.scatter(x, z, c=colors, s=30, alpha=0.7)
            ax3.plot(x, z, 'k-', alpha=0.3, linewidth=1)

            # Plot triangle in XZ plane
            for e in edges:
                ax3.plot(
                    [vertices[e[0]][0], vertices[e[1]][0]],
                    [vertices[e[0]][2], vertices[e[1]][2]],
                    color=triangle_color, linewidth=2, alpha=0.7
                )

            # Fill triangle area in XZ plane with a Polygon
            xz_vertices = vertices[:, [0, 2]]  # Extract x,z coordinates
            print(f"XZ vertices: {xz_vertices}")
            try:
                tri_xz = Polygon(xz_vertices, alpha=transparency, color=triangle_color)
                ax3.add_patch(tri_xz)
            except Exception as e:
                print(f"Could not create XZ triangle: {e}")

            # Add start and end points
            ax3.scatter(x[0], z[0], color='green', s=80, label='Start', edgecolor='black')
            ax3.scatter(x[-1], z[-1], color='red', s=80, label='End', edgecolor='black')

            ax3.set_xlabel('X', fontweight='bold')
            ax3.set_ylabel('Z', fontweight='bold')
            ax3.set_title('Front View (XZ Plane)', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # YZ Plane (Side View)
            ax4 = fig.add_subplot(224)
            ax4.set_facecolor('#f8f8f8')
            ax4.scatter(y, z, c=colors, s=30, alpha=0.7)
            ax4.plot(y, z, 'k-', alpha=0.3, linewidth=1)

            # Plot triangle in YZ plane
            for e in edges:
                ax4.plot(
                    [vertices[e[0]][1], vertices[e[1]][1]],
                    [vertices[e[0]][2], vertices[e[1]][2]],
                    color=triangle_color, linewidth=2, alpha=0.7
                )

            # Fill triangle area in YZ plane with a Polygon
            yz_vertices = vertices[:, [1, 2]]  # Extract y,z coordinates
            print(f"YZ vertices: {yz_vertices}")
            try:
                tri_yz = Polygon(yz_vertices, alpha=transparency, color=triangle_color)
                ax4.add_patch(tri_yz)
            except Exception as e:
                print(f"Could not create YZ triangle: {e}")

            # Add start and end points
            ax4.scatter(y[0], z[0], color='green', s=80, label='Start', edgecolor='black')
            ax4.scatter(y[-1], z[-1], color='red', s=80, label='End', edgecolor='black')

            ax4.set_xlabel('Y', fontweight='bold')
            ax4.set_ylabel('Z', fontweight='bold')
            ax4.set_title('Side View (YZ Plane)', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # Add colorbar for time progression
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3, ax4], orientation='horizontal',
                                pad=0.05, aspect=40, shrink=0.6)
            cbar.set_label('Frame Progression', fontweight='bold')

            # Add triangle information as text
            fig.text(0.5, 0.01,
                     f"Reference Triangle - Position: {triangle_info['position']}, Size: {triangle_info['size']:.2f}, " +
                     f"Orientation: {triangle_info['orientation']}",
                     ha='center', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

            # Add overall title
            plt.suptitle('Trajectory Analysis with Reference Triangle', fontsize=16, fontweight='bold', y=0.98)

            plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Make room for title and footer

            # Save if output file provided
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Visualization with reference triangle saved to {output_file}")

            plt.show()
            plt.close(fig)

            return True

        except Exception as e:
            print(f"Error creating visualization with reference triangle: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_with_reference_triangle(self, triangle_params=None, base_filename=None):
        """
        Create and save visualizations with a reference triangle

        Args:
            triangle_params: Parameters for reference triangle
            base_filename: Base name for output files

        Returns:
            dict: Paths to saved files
        """
        if not self.trajectory_3d:
            print("No trajectory data available")
            return {}

        try:
            # Generate base filename if not provided
            if base_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"trajectory_with_triangle_{timestamp}"
            else:
                # Strip extension if provided
                base_filename = os.path.splitext(base_filename)[0]

            # Create reference triangle
            triangle_info = self.create_reference_triangle(triangle_params)

            # Save visualization of original trajectory with triangle
            orig_output = f"{base_filename}_original.png"
            self.visualize_with_reference_triangle(
                self.trajectory_3d,
                triangle_info,
                rotated=False,
                output_file=orig_output
            )

            # Get rotated trajectory if available
            rotated_trajectory = None
            rotated_output = None

            # Try to rotate the trajectory if not already done
            if not hasattr(self, 'rotated_trajectory') or self.rotated_trajectory is None:
                try:
                    self.rotated_trajectory, self.rotation_info = self.rotate_trajectory_to_xy_plane()
                except Exception as e:
                    print(f"Could not rotate trajectory: {e}")

            # Save visualization of rotated trajectory with triangle if available
            if hasattr(self, 'rotated_trajectory') and self.rotated_trajectory:
                rotated_output = f"{base_filename}_rotated.png"
                self.visualize_with_reference_triangle(
                    self.rotated_trajectory,
                    triangle_info,
                    rotated=True,
                    output_file=rotated_output
                )

            # Save triangle information
            triangle_file = f"{base_filename}_triangle.json"
            with open(triangle_file, 'w') as f:
                json.dump(triangle_info, f, indent=4, cls=NumpyEncoder)

            print(f"Triangle information saved to {triangle_file}")

            # Return paths to saved files
            result = {
                'original': orig_output,
                'triangle_info': triangle_file
            }

            if rotated_output:
                result['rotated'] = rotated_output

            return result

        except Exception as e:
            print(f"Error saving with reference triangle: {e}")
            import traceback
            traceback.print_exc()
            return {}


if __name__ == '__main__':
    print("\n=======================================")
    print("  EXCLUSIVE CAMERA TRACKER - 3D TRACKING")
    print("=======================================\n")

    # Define paths to your videos. Change these if the files are stored elsewhere.
    front_video_path = "front7.mp4"  # Or "videos/front7.mp4" if in a subfolder
    side_video_path = "side7.mp4"
    floor_video_path = "flor7.mp4"  # Notice the filename if different from 'floor7.mp4'

    print(f"Front camera video: {front_video_path}")
    print(f"Side camera video: {side_video_path}")
    print(f"Floor camera video: {floor_video_path}")

    # Create output directory if it doesn't exist
    output_dir = "tracking_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Session timestamp: {timestamp}")

    # Create an instance of the tracker with the video paths
    print("\nInitializing 3D tracker...")
    tracker = ExclusiveCameraTracker(front_video_path, side_video_path, floor_video_path)

    # Run the tracker
    print("\nStarting tracking. Press ESC to stop...")
    tracker.run()

    # After tracking is complete, save results
    if len(tracker.trajectory_3d) > 0:
        print("\n=======================================")
        print("  SAVING TRACKING RESULTS")
        print("=======================================\n")

        # Base filename for all outputs
        base_filename = f"{output_dir}/tracking_{timestamp}"

        # Save trajectory data
        print("Saving trajectory data...")
        saved_files = tracker.save_trajectory(base_filename)

        # Print information about saved files
        if saved_files:
            print("\nThe following files have been saved:")
            for file_type, filepath in saved_files.items():
                print(f"  - {file_type.upper()}: {filepath}")

        # Export camera calibration
        print("\nExporting camera calibration settings...")
        calibration_file = tracker.export_camera_calibration(f"{base_filename}_calibration")
        if calibration_file:
            print(f"  - CALIBRATION: {calibration_file}")

        # Generate HTML report
        print("\nGenerating comprehensive HTML report...")
        report_file = tracker.generate_report(f"{output_dir}/report_{timestamp}.html")
        if report_file:
            print(f"  - REPORT: {report_file}")

        # ROTATION: Perform trajectory rotation to XY plane
        print("\n=======================================")
        print("  ROTATING TRAJECTORY TO XY PLANE")
        print("=======================================\n")

        # Store the rotated trajectory for later use
        tracker.rotated_trajectory, tracker.rotation_info = tracker.rotate_trajectory_to_xy_plane()

        if tracker.rotation_info["success"]:
            # Save the rotated trajectory
            print("\nSaving rotated trajectory data...")
            rotated_files = tracker.save_rotated_trajectory(
                tracker.rotated_trajectory,
                tracker.rotation_info,
                f"{base_filename}_rotated"
            )

            # Print information about saved rotated files
            if rotated_files:
                print("\nThe following rotated trajectory files have been saved:")
                for file_type, filepath in rotated_files.items():
                    print(f"  - ROTATED_{file_type.upper()}: {filepath}")

            # Add rotation information to HTML report by adding a new section
            print("\nNote: View the rotated_viz.png file to see the rotated trajectory visualization")
        else:
            print(f"Could not rotate trajectory: {tracker.rotation_info.get('reason', 'Unknown error')}")

        # Create reference triangle
        print("\n=======================================")
        print("  TRAJECTORY ANALYSIS FOR TRIANGLE PLACEMENT")
        print("=======================================")

        # Manual triangle with precise control
        # Important: Use coordinates appropriate for your specific trajectory
        # These coordinates are just examples and need to be adjusted
        manual_triangle_params = {
            'color': 'magenta',
            'transparency': 0.4,
            'vertices': [
                [43, 91, 47],  # First point - ADJUST FOR YOUR TRAJECTORY
                [50, 101, 47],  # Second point - ADJUST FOR YOUR TRAJECTORY
                [51, 84, 47]  # Third point - ADJUST FOR YOUR TRAJECTORY
            ]
        }

        print("\nCreating manually positioned triangle...")
        triangle_files = tracker.save_with_reference_triangle(
            triangle_params=manual_triangle_params,
            base_filename=f"{base_filename}_triangle"
        )

        if triangle_files:
            print("\nReference triangle files have been saved:")
            for key, filepath in triangle_files.items():
                print(f"  - TRIANGLE_{key.upper()}: {filepath}")

        # Display trajectory analysis
        print("\n=======================================")
        print("  TRAJECTORY ANALYSIS SUMMARY")
        print("=======================================")
        stats = tracker.analyze_trajectory()

        # Print a more user-friendly summary
        if stats:
            if 'primary_plane' in stats:
                print(f"\nPrimary motion detected in the {stats['primary_plane']}")
                if stats.get('is_mostly_2d', False):
                    print("Motion is primarily two-dimensional")
                else:
                    print("Motion has significant three-dimensional components")

            print(f"\nTotal distance traveled: {stats.get('total_distance', 0):.2f} units")
            print(f"Path efficiency: {stats.get('path_efficiency', 0):.2f} (1.0 is a straight line)")

            # Print camera confidence
            if 'avg_front_confidence' in stats:
                print("\nCamera detection quality:")
                print(f"  Front camera: {stats['avg_front_confidence']:.2f}")
                print(f"  Side camera:  {stats['avg_side_confidence']:.2f}")
                print(f"  Floor camera: {stats['avg_floor_confidence']:.2f}")

            # Print Y calibration info
            if 'y_calibration_method' in stats:
                print(f"\nY-coordinate calibration method: {stats['y_calibration_method']}")
                print(f"Front camera Y direction: {stats.get('final_front_y_direction', 1)}")
                print(f"Side camera Y direction:  {stats.get('final_side_y_direction', 1)}")

                if 'avg_y_agreement' in stats:
                    agreement = stats['avg_y_agreement']
                    if agreement > 0.8:
                        agreement_msg = "Excellent agreement"
                    elif agreement > 0.6:
                        agreement_msg = "Good agreement"
                    elif agreement > 0.4:
                        agreement_msg = "Fair agreement"
                    else:
                        agreement_msg = "Poor agreement"

                    print(f"Camera Y-coordinate agreement: {agreement:.2f} ({agreement_msg})")

        # Print info about the rotation if successful
        if hasattr(tracker, 'rotation_info') and tracker.rotation_info.get("success", False):
            print("\n=======================================")
            print("  ROTATION ANALYSIS")
            print("=======================================")

            print(f"\nOriginal primary plane: {tracker.rotation_info.get('primary_dimensions_before', 'Unknown')}")
            print(f"Rotated to: {tracker.rotation_info.get('primary_dimensions_after', 'XY plane')}")

            alignment = tracker.rotation_info.get('alignment_quality', 0)
            print(f"Alignment quality: {alignment:.4f} (higher is better)")

            if alignment > 0.95:
                print("Excellent alignment with XY plane achieved")
            elif alignment > 0.8:
                print("Good alignment with XY plane achieved")
            elif alignment > 0.6:
                print("Moderate alignment with XY plane achieved")
            else:
                print("Limited alignment with XY plane achieved")

            if 'rotated_variance' in tracker.rotation_info:
                rot_var = tracker.rotation_info['rotated_variance']
                xy_var = rot_var[0] + rot_var[1]
                z_var = rot_var[2]
                print(f"\nAfter rotation:")
                print(f"  Variance in XY plane: {xy_var:.2f}")
                print(f"  Variance in Z direction: {z_var:.2f}")
                print(f"  Ratio: {xy_var / (z_var + 1e-10):.2f}:1 (higher means better XY plane alignment)")

        print("\n=======================================")
        print("  TRIANGLE PLACEMENT INSTRUCTIONS")
        print("=======================================")

        print("\nTo get precise triangle placement:")
        print("1. Look at the saved triangle visualization")
        print("2. Adjust the coordinates in the 'vertices' list in the 'manual_triangle_params' dictionary")
        print("3. Run the code again to see the updated triangle position")
        print("4. Repeat until the triangle is positioned exactly where you want it")

        print("\n=======================================")
        print(f"All results saved to: {os.path.abspath(output_dir)}")
        print("=======================================\n")

    else:
        print("\nNo trajectory data was recorded. Check video files and camera setup.")

    print("Tracking session complete.")