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
                avg_correlation = sum(self.y_correlation_scores) / len(self.y_correlation_scores)
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
        return None @ staticmethod


def load_camera_calibration(filename):
    """
    Load camera calibration settings from a JSON file

    Args:
        filename: Path to the calibration file

    Returns:
        dict: Configuration dictionary that can be passed to the constructor
    """
    try:
        with open(filename, 'r') as f:
            calibration_data = json.load(f)

        # Extract camera settings into a config dictionary
        config = {
            'camera_flip': {
                'front_x': calibration_data['front_camera']['flip_x'],
                'front_y': calibration_data['front_camera']['flip_y'],
                'side_x': calibration_data['side_camera']['flip_x'],
                'side_y': calibration_data['side_camera']['flip_y'],
                'floor_x': calibration_data['floor_camera']['flip_x'],
                'floor_y': calibration_data['floor_camera']['flip_y']
            },
            # Disable startup calibration if loading from file
            'enable_startup_calibration': False,
            # Disable cross-camera analysis if loading from file
            'use_cross_camera_analysis': False,
            # Use manual mode since we're loading pre-determined settings
            'y_calibration_mode': 'manual'
        }

        print(f"Loaded camera calibration from {filename}")
        print(f"Front camera Y direction: {calibration_data['front_camera']['y_direction']}")
        print(f"Side camera Y direction: {calibration_data['side_camera']['y_direction']}")

        return config, calibration_data
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return None, None


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
        ax1.scatter(x_points[0], y_points[0], z_points[0], color='green', s=100, label='Start', edgecolor='black')
        ax1.scatter(x_points[-1], y_points[-1], z_points[-1], color='red', s=100, label='End', edgecolor='black')

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
        ax2.scatter(x_points[0], y_points[0], color='green', s=100, marker='o', label='Start', edgecolor='black')
        ax2.scatter(x_points[-1], y_points[-1], color='red', s=100, marker='o', label='End', edgecolor='black')
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
        ax3.scatter(x_points[0], z_points[0], color='green', s=100, marker='o', label='Start', edgecolor='black')
        ax3.scatter(x_points[-1], z_points[-1], color='red', s=100, marker='o', label='End', edgecolor='black')
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
            ax4.plot(frames_y, display_front_y, 'g-', label=f'Front Y (Dir: {self.front_y_direction})', linewidth=2)
            ax4.plot(frames_y, display_side_y, 'm-', label=f'Side Y (Dir: {self.side_y_direction})', linewidth=2)

            # Plot agreement scores if available
            if self.y_agreement_scores and len(self.y_agreement_scores) >= display_start:
                display_agreement = self.y_agreement_scores[display_start:display_start + window_size]
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
        fig.text(0.5, 0.01, f"Total Frames: {self.frame_count}   Points Tracked: {len(self.trajectory_3d)}   "
                            f"Front Y-Dir: {self.front_y_direction}   Side Y-Dir: {self.side_y_direction}",
                 ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

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

        def save_trajectory(self, base_filename=None):
    """
    Save the tracked trajectory data and analysis

    Args:
        base_filename: Base name for saved files (without extension)
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
            "calibration_method": self.calibration_results.get("method", "standard") if hasattr(self,
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
            "calibration_results": self.calibration_results if hasattr(self, 'calibration_results') else {},
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
            stats['avg_x_movement'] = sum(self.dimension_movements['X']) / len(self.dimension_movements['X'])
            stats['avg_y_movement'] = sum(self.dimension_movements['Y']) / len(self.dimension_movements['Y'])
            stats['avg_z_movement'] = sum(self.dimension_movements['Z']) / len(self.dimension_movements['Z'])

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
            stats['is_mostly_2d'] = stats['dimensionality_ratio'] < 0.2  # If < 20% movement in perpendicular axis

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
                stats['avg_y_correlation'] = sum(self.y_correlation_scores) / len(self.y_correlation_scores)

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
        values = self.front_y_values[-num_frames:] if len(self.front_y_values) > num_frames else self.front_y_values
        scale_x = gw / (num_frames - 1) if num_frames > 1 else 0
        scale_y = gh

        for i in range(1, len(values)):
            x1 = gx + int((i - 1) * scale_x)
            y1 = gy + int((1 - values[i - 1]) * scale_y)
            x2 = gx + int(i * scale_x)
            y2 = gy + int((1 - values[i]) * scale_y)
            cv2.line(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Plot side camera values (magenta)
        values = self.side_y_values[-num_frames:] if len(self.side_y_values) > num_frames else self.side_y_values

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
                    front_y, side_y, agreement, correlation = self.verify_y_coordinate_directions(front_point,
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

                    cv2.rectangle(front_resized, (50, disp_height - 50), (50 + bar_width, disp_height - 40), (0, 0, 0),
                                  1)
                    cv2.rectangle(front_resized, (50, disp_height - 50), (50 + filled_width, disp_height - 40),
                                  (0, 255, 0), -1)

                    cv2.rectangle(side_resized, (50, disp_height - 50), (50 + bar_width, disp_height - 40), (0, 0, 0),
                                  1)
                    cv2.rectangle(side_resized, (50, disp_height - 50), (50 + filled_width, disp_height - 40),
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
                    cv2.putText(front_resized, f"Y: {self.front_y_direction}", (disp_width - 100, disp_height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw Y direction arrow on side camera (X in side view)
                    arrow_start = (disp_width - 60, disp_height - 40)
                    if self.side_y_direction > 0:
                        arrow_end = (disp_width - 100, disp_height - 40)
                    else:
                        arrow_end = (disp_width - 20, disp_height - 40)
                    cv2.arrowedLine(side_resized, arrow_start, arrow_end, (255, 0, 255), 3, tipLength=0.3)
                    cv2.putText(side_resized, f"Y: {self.side_y_direction}", (disp_width - 100, disp_height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    # Add verification status
                    status_text = self.y_verification_state
                    if len(status_text) > 25:  # Truncate if too long
                        status_text = status_text[:25] + "..."

                    status_color = (0, 255, 0) if "agree" in self.y_verification_state.lower() else (0, 0, 255)
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
                comparison_vis = self.create_y_comparison_visualization(front_point, side_point, front_frame,
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
        return Falseimport
        cv2


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import os
from scipy.ndimage import gaussian_filter1d
from datetime import datetime


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
            front_video_path: Path to front camera video
            side_video_path: Path to side camera video
            floor_video_path: Path to floor camera video
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

        # Check if videos opened successfully
        if not all([self.cap_front.isOpened(), self.cap_side.isOpened(), self.cap_floor.isOpened()]):
            raise ValueError("Could not open one or more video files")

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
                    floor_result = self.detect_red_object(floor_floor, 'floor')

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