#!/usr/bin/env python3
"""
rviz_paths_node.py

Publishes DECK-GA paths + all target points to RViz as MarkerArray.

Coordinate mapping (MUST match deckga_execute.py):
- x_cmd = x_raw * scale_xy
- y_cmd = y_raw * scale_xy
- z_cmd = max(z_min, z_raw * scale_z)

Usage example:
  python3 deckga_ros2/rviz_paths_node.py \
    --frame earth \
    --topic /deckga/markers_seed11 \
    --points_pkl data/points/points_seed11_n30.pkl \
    --deckga_pkl deckga_ros2/data/deckga_output.pkl \
    --scale_xy 0.05 --scale_z 0.05 \
    --z_min 1.0
"""

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def load_pkl_xyz(p: Path) -> np.ndarray:
    with p.open("rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{p} must be Nx3, got {arr.shape}")
    return arr


class DeckgaRvizMarkers(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("deckga_rviz_markers")

        # Frame + topic
        self.frame_id = args.frame
        self.topic = args.topic

        # Scaling + altitude mapping
        self.scale_xy = float(args.scale_xy)
        self.scale_z = float(args.scale_z)
        self.z_min = float(args.z_min)

        # Marker sizes (tunable)
        self.point_sphere_d = float(args.point_sphere_d)
        self.wp_sphere_d = float(args.wp_sphere_d)
        self.line_width = float(args.line_width)

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL  # latched markers
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.pub = self.create_publisher(MarkerArray, self.topic, qos)

        repo_root = Path(__file__).resolve().parents[1]
        self.points_pkl = (repo_root / args.points_pkl).resolve()
        self.deckga_pkl = (repo_root / args.deckga_pkl).resolve()

        self.get_logger().info(f"Frame: {self.frame_id}")
        self.get_logger().info(f"Publishing MarkerArray on: {self.topic}")
        self.get_logger().info(f"Points: {self.points_pkl}")
        self.get_logger().info(f"DeckGA:  {self.deckga_pkl}")
        self.get_logger().info(
            f"Map: scale_xy={self.scale_xy}, scale_z={self.scale_z}, z_min={self.z_min}"
        )

        # Load points
        self.points = load_pkl_xyz(self.points_pkl)

        # Load deckga paths
        with self.deckga_pkl.open("rb") as f:
            data = pickle.load(f)

        raw_paths = data.get("deckga_paths", [])
        self.paths: List[np.ndarray] = [np.asarray(p, dtype=float) for p in raw_paths]

        self.get_logger().info(f"Loaded points: {len(self.points)}")
        self.get_logger().info(f"Loaded deckga_paths: {len(self.paths)} UAV routes")

        # Publish repeatedly so RViz always catches it
        self.timer = self.create_timer(1.0 / float(args.rate_hz), self.publish)

    def _map_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """
        Apply the same mapping used in deckga_execute.py:
          x = x*scale_xy
          y = y*scale_xy
          z = max(z_min, z*scale_z)
        """
        out = np.asarray(xyz, dtype=float).copy()
        out[:, 0] *= self.scale_xy
        out[:, 1] *= self.scale_xy
        out[:, 2] *= self.scale_z
        out[:, 2] = np.maximum(out[:, 2], self.z_min)
        return out

    def publish(self):
        ma = MarkerArray()

        # 1) All target waypoints (white spheres)
        pts = self._map_xyz(self.points)
        m_points = Marker()
        m_points.header.frame_id = self.frame_id
        m_points.header.stamp = self.get_clock().now().to_msg()
        m_points.ns = "deckga_points"
        m_points.id = 0
        m_points.type = Marker.SPHERE_LIST
        m_points.action = Marker.ADD
        m_points.scale.x = self.point_sphere_d
        m_points.scale.y = self.point_sphere_d
        m_points.scale.z = self.point_sphere_d
        m_points.color.a = 1.0
        m_points.color.r = 1.0
        m_points.color.g = 1.0
        m_points.color.b = 1.0
        m_points.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]
        ma.markers.append(m_points)

        # 2) Each UAV path (colored line strips + colored waypoint spheres)
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

        for i, p in enumerate(self.paths):
            if p is None or len(p) == 0:
                continue

            p = np.asarray(p, dtype=float)

            # Remove repeated final start (common in GA output)
            if len(p) >= 2 and np.allclose(p[0], p[-1]):
                p = p[:-1]

            p_mapped = self._map_xyz(p)

            r, g, b = colors[i % len(colors)]

            # Line strip
            m_line = Marker()
            m_line.header.frame_id = self.frame_id
            m_line.header.stamp = self.get_clock().now().to_msg()
            m_line.ns = f"path_drone{i}"
            m_line.id = 100 + i
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.scale.x = self.line_width
            m_line.color.a = 1.0
            m_line.color.r = float(r)
            m_line.color.g = float(g)
            m_line.color.b = float(b)
            m_line.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in p_mapped]
            ma.markers.append(m_line)

            # Waypoints along that path
            m_wp = Marker()
            m_wp.header.frame_id = self.frame_id
            m_wp.header.stamp = self.get_clock().now().to_msg()
            m_wp.ns = f"wps_drone{i}"
            m_wp.id = 200 + i
            m_wp.type = Marker.SPHERE_LIST
            m_wp.action = Marker.ADD
            m_wp.scale.x = self.wp_sphere_d
            m_wp.scale.y = self.wp_sphere_d
            m_wp.scale.z = self.wp_sphere_d
            m_wp.color.a = 1.0
            m_wp.color.r = float(r)
            m_wp.color.g = float(g)
            m_wp.color.b = float(b)
            m_wp.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in p_mapped]
            ma.markers.append(m_wp)

        self.pub.publish(ma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/deckga/markers")
    ap.add_argument("--frame", default="map")

    # Must match deckga_execute.py mapping
    ap.add_argument("--scale_xy", type=float, default=0.05)
    ap.add_argument("--scale_z", type=float, default=0.05)
    ap.add_argument("--z_min", type=float, default=1.0)

    ap.add_argument("--rate_hz", type=float, default=2.0)

    # Marker geometry tuning
    ap.add_argument("--point_sphere_d", type=float, default=0.12)  # all points (white)
    ap.add_argument("--wp_sphere_d", type=float, default=0.08)     # per-path waypoints
    ap.add_argument("--line_width", type=float, default=0.05)

    ap.add_argument("--points_pkl", default="data/points/points_current.pkl")
    ap.add_argument("--deckga_pkl", default="deckga_ros2/data/deckga_output.pkl")
    args = ap.parse_args()

    rclpy.init()
    node = DeckgaRvizMarkers(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
