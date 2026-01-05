#!/usr/bin/env python3
# rviz_obstacles_node.py
#
# Visualize DECK-GA + QuickNav obstacles and paths in RViz using MarkerArray.
# - Reads the same deckga_quicknav_output.pkl as deckga_as2_execute.py
# - Uses the same SCALE and MIN_Z so markers match the flown trajectories.

import pickle
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray


# Must match your flight script
SCALE = 0.05
MIN_Z = 2.0

# Relative path inside DECKGA_QuickNav_3D repo
DATA_FILE = Path(__file__).resolve().parent / "data" / "deckga_quicknav_output.pkl"


class DeckgaObstaclesViz(Node):
    def __init__(self):
        super().__init__("deckga_obstacles_viz")

        self.get_logger().info("Loading DECK-GA QuickNav data for RViz markers...")

        # Load pickle data
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)

        # Paths: list of [N_i x 3] arrays/lists
        self.raw_paths = data["quicknav_paths"]

        # Obstacles: positions + sizes (shape may be [N,3] or [N])
        self.obstacle_xyz = np.array(data["obstacle_xyz"], dtype=float)
        self.obstacle_size = np.array(data["obs_size"], dtype=float)

        # Sanitize & scale paths the same way as deckga_as2_execute.py
        self.paths = [self._sanitize_and_scale_path(p) for p in self.raw_paths]

        self.get_logger().info(
            f"Loaded {len(self.paths)} UAV paths and "
            f"{self.obstacle_xyz.shape[0]} obstacles."
        )

        # Publisher for MarkerArray
        self.marker_pub = self.create_publisher(MarkerArray, "deckga_markers", 10)

        # Publish at 1 Hz
        self.timer = self.create_timer(1.0, self.timer_callback)

    # ------------------------------------------------------------------ #
    # HELPER FUNCTIONS
    # ------------------------------------------------------------------ #

    def _sanitize_and_scale_path(self, path):
        """Apply SCALE and MIN_Z exactly like the flight script."""
        out = []
        for p in path:
            x = float(p[0]) * SCALE
            y = float(p[1]) * SCALE
            z = float(p[2]) * SCALE
            if z < MIN_Z:
                z = MIN_Z
            out.append((x, y, z))
        return out

    def _make_obstacle_markers(self):
        """Create cube markers for each obstacle (red)."""
        markers = []
        now = self.get_clock().now().to_msg()

        # Scale positions and sizes
        pos_scaled = self.obstacle_xyz * SCALE

        # Handle different shapes for size (N,) or (N,3)
        size_scaled = self.obstacle_size * SCALE

        for i, pos in enumerate(pos_scaled):
            m = Marker()
            m.header.frame_id = "earth"  # same as your go_to frame
            m.header.stamp = now
            m.ns = "deckga_obstacles"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            # Obstacles can sit on ground or at their z; we keep z scaled:
            m.pose.position.z = float(pos[2])

            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 1.0

            if size_scaled.ndim == 1:
                s = float(size_scaled[i])
                m.scale.x = s
                m.scale.y = s
                m.scale.z = s
            else:
                sx, sy, sz = size_scaled[i]
                m.scale.x = float(sx)
                m.scale.y = float(sy)
                m.scale.z = float(sz)

            # Red, semi-transparent
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.6

            m.lifetime.sec = 0  # 0 = forever (until overwritten)

            markers.append(m)

        return markers

    def _make_path_markers(self):
        """Create sphere markers for each UAV waypoint."""
        markers = []
        now = self.get_clock().now().to_msg()

        # Colors per UAV: 0=blue, 1=green, 2=yellow
        colors = [
            (0.0, 0.2, 1.0),  # UAV0: blue-ish
            (0.0, 0.8, 0.0),  # UAV1: green
            (1.0, 0.8, 0.0),  # UAV2: yellow
        ]

        id_offset = 100  # to avoid id collision with obstacles

        for uav_idx, path in enumerate(self.paths):
            if uav_idx >= len(colors):
                # If more than 3 UAVs appear, just skip extra or reuse colors
                color = (1.0, 1.0, 1.0)
            else:
                color = colors[uav_idx]

            for k, (x, y, z) in enumerate(path):
                m = Marker()
                m.header.frame_id = "earth"
                m.header.stamp = now
                m.ns = f"uav{uav_idx}_path"
                # Unique ID per waypoint per UAV
                m.id = id_offset * (uav_idx + 1) + k
                m.type = Marker.SPHERE
                m.action = Marker.ADD

                m.pose.position.x = x
                m.pose.position.y = y
                m.pose.position.z = z

                m.pose.orientation.x = 0.0
                m.pose.orientation.y = 0.0
                m.pose.orientation.z = 0.0
                m.pose.orientation.w = 1.0

                m.scale.x = 0.15
                m.scale.y = 0.15
                m.scale.z = 0.15

                m.color.r = color[0]
                m.color.g = color[1]
                m.color.b = color[2]
                m.color.a = 0.9

                m.lifetime.sec = 0

                markers.append(m)

        return markers

    # ------------------------------------------------------------------ #
    # TIMER CALLBACK
    # ------------------------------------------------------------------ #

    def timer_callback(self):
        """Publish MarkerArray periodically."""
        marker_array = MarkerArray()
        marker_array.markers.extend(self._make_obstacle_markers())
        marker_array.markers.extend(self._make_path_markers())

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = DeckgaObstaclesViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
