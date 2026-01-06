#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

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


def clamp_min_z(xyz: np.ndarray, z_min: float) -> np.ndarray:
    out = np.asarray(xyz, dtype=float).copy()
    out[:, 2] = np.maximum(out[:, 2], z_min)
    return out


class DeckgaRvizMarkers(Node):
    def __init__(self, args):
        super().__init__("deckga_rviz_markers")

        self.frame_id = args.frame
        self.scale_xy = args.scale_xy
        self.scale_z = args.scale_z
        self.z_min = args.z_min

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL  # "latched" markers
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.pub = self.create_publisher(MarkerArray, args.topic, qos)

        repo_root = Path(__file__).resolve().parents[1]
        self.points_pkl = repo_root / args.points_pkl
        self.deckga_pkl = repo_root / args.deckga_pkl

        self.get_logger().info(f"Frame: {self.frame_id}")
        self.get_logger().info(f"Publishing MarkerArray on: {args.topic}")
        self.get_logger().info(f"Points: {self.points_pkl}")
        self.get_logger().info(f"DeckGA:  {self.deckga_pkl}")

        self.points = load_pkl_xyz(self.points_pkl)

        with self.deckga_pkl.open("rb") as f:
            data = pickle.load(f)

        self.paths = [np.asarray(p, dtype=float) for p in data["deckga_paths"]]

        self.get_logger().info(f"Loaded points: {len(self.points)}")
        self.get_logger().info(f"Loaded deckga_paths: {len(self.paths)} UAV routes")

        # publish repeatedly so RViz always catches it
        self.timer = self.create_timer(1.0 / args.rate_hz, self.publish)

    def _scale_xyz(self, xyz: np.ndarray) -> np.ndarray:
        out = np.asarray(xyz, dtype=float).copy()
        out[:, 0] *= self.scale_xy
        out[:, 1] *= self.scale_xy
        out[:, 2] *= self.scale_z
        out = clamp_min_z(out, self.z_min)
        return out

    def publish(self):
        ma = MarkerArray()

        # 1) All target waypoints (white spheres)
        pts = self._scale_xyz(self.points)
        m_points = Marker()
        m_points.header.frame_id = self.frame_id
        m_points.header.stamp = self.get_clock().now().to_msg()
        m_points.ns = "deckga_points"
        m_points.id = 0
        m_points.type = Marker.SPHERE_LIST
        m_points.action = Marker.ADD
        m_points.scale.x = 0.12
        m_points.scale.y = 0.12
        m_points.scale.z = 0.12
        m_points.color.a = 1.0
        m_points.color.r = 1.0
        m_points.color.g = 1.0
        m_points.color.b = 1.0
        m_points.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]
        ma.markers.append(m_points)

        # 2) Each UAV path (colored line strips + small spheres)
        for i, p in enumerate(self.paths):
            p = np.asarray(p, dtype=float)

            # remove repeated final start (common in GA output)
            if len(p) >= 2 and np.allclose(p[0], p[-1]):
                p = p[:-1]

            p = self._scale_xyz(p)

            # Line strip
            m_line = Marker()
            m_line.header.frame_id = self.frame_id
            m_line.header.stamp = self.get_clock().now().to_msg()
            m_line.ns = f"path_drone{i}"
            m_line.id = 100 + i
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.scale.x = 0.05
            m_line.color.a = 1.0

            # simple distinct colors
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
            r, g, b = colors[i % len(colors)]
            m_line.color.r = float(r)
            m_line.color.g = float(g)
            m_line.color.b = float(b)

            m_line.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in p]
            ma.markers.append(m_line)

            # Waypoints along that path
            m_wp = Marker()
            m_wp.header.frame_id = self.frame_id
            m_wp.header.stamp = self.get_clock().now().to_msg()
            m_wp.ns = f"wps_drone{i}"
            m_wp.id = 200 + i
            m_wp.type = Marker.SPHERE_LIST
            m_wp.action = Marker.ADD
            m_wp.scale.x = 0.08
            m_wp.scale.y = 0.08
            m_wp.scale.z = 0.08
            m_wp.color.a = 1.0
            m_wp.color.r = float(r)
            m_wp.color.g = float(g)
            m_wp.color.b = float(b)
            m_wp.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in p]
            ma.markers.append(m_wp)

        self.pub.publish(ma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/deckga/markers")
    ap.add_argument("--frame", default="map")

    # IMPORTANT: match deckga_execute scaling (your earlier flights indicate xy was /20 => 0.05)
    ap.add_argument("--scale_xy", type=float, default=0.05)
    ap.add_argument("--scale_z", type=float, default=0.05)
    ap.add_argument("--z_min", type=float, default=2.0)

    ap.add_argument("--rate_hz", type=float, default=2.0)

    ap.add_argument("--points_pkl", default="data/points/points_current.pkl")
    ap.add_argument("--deckga_pkl", default="deckga_ros2/data/deckga_output.pkl")
    args = ap.parse_args()

    rclpy.init()
    node = DeckgaRvizMarkers(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
