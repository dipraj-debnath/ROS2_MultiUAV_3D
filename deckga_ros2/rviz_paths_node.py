#!/usr/bin/env python3
"""
rviz_paths_node.py

RViz MarkerArray publisher for DECK-GA routes.

Critical requirement:
- This node MUST apply the exact same coordinate transform as deckga_execute.py.

Transform pipeline (must match deckga_execute.py):
    (A) Optional unshift (subtract offset_used) depending on coord_mode
    (B) Scaling: XY and Z
    (C) Z offset and Z minimum clamp

This version hardcodes the defaults you want so you can run only:
    python3 rviz_paths_node.py
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


# =========================
# Hardcoded defaults (match execution)
# =========================

DEFAULT_DECKGA_PKL = "/home/dipraj/Documents/GitHub/ROS2_MultiUAV_3D/deckga_ros2/data/deckga_output.pkl"

DEFAULT_FRAME_ID = "earth"
DEFAULT_TOPIC = "/deckga/markers"

DEFAULT_COORD_MODE = "original"
DEFAULT_SCALE_XY = 0.05
DEFAULT_SCALE_Z = 0.05
DEFAULT_Z_OFFSET = 2.5
DEFAULT_Z_MIN = 3.0


# =========================
# Transform (shared logic)
# =========================

def _stack_all_points(paths: Sequence[np.ndarray]) -> np.ndarray:
    pts: List[np.ndarray] = []
    for p in paths:
        if p.size == 0:
            continue
        pts.append(p)
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pts)


def _auto_is_shifted(all_pts: np.ndarray, offset_used: Optional[np.ndarray]) -> bool:
    if offset_used is None or all_pts.shape[0] == 0:
        return False

    min_xy = all_pts[:, :2].min(axis=0)
    mean_xy = all_pts[:, :2].mean(axis=0)

    if min_xy[0] < -1e-6 or min_xy[1] < -1e-6:
        return False

    return (mean_xy[0] >= 0.20 * offset_used[0]) or (mean_xy[1] >= 0.20 * offset_used[1])


@dataclass(frozen=True)
class TransformConfig:
    coord_mode: str
    scale_xy: float
    scale_z: float
    z_offset: float
    z_min: float


def transform_paths(
    raw_paths: Sequence[Any],
    offset_used: Optional[np.ndarray],
    tf: TransformConfig,
) -> List[np.ndarray]:
    paths_np: List[np.ndarray] = []
    for p in raw_paths:
        arr = np.asarray(p, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Each path must be Nx3. Got shape {arr.shape}")
        paths_np.append(arr)

    all_pts = _stack_all_points(paths_np)

    if tf.coord_mode == "shifted":
        do_unshift = True
    elif tf.coord_mode == "original":
        do_unshift = False
    else:
        do_unshift = _auto_is_shifted(all_pts, offset_used)

    out: List[np.ndarray] = []
    for arr in paths_np:
        a = arr.copy()

        if do_unshift and offset_used is not None:
            a = a - offset_used

        a[:, 0] *= float(tf.scale_xy)
        a[:, 1] *= float(tf.scale_xy)
        a[:, 2] *= float(tf.scale_z)

        a[:, 2] += float(tf.z_offset)
        if float(tf.z_min) > 0.0:
            a[:, 2] = np.maximum(a[:, 2], float(tf.z_min))

        out.append(a)

    return out


def enforce_closed_tour(path: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Ensure the tour ends at the first waypoint (so RViz + execution match)."""
    if path.size == 0:
        return path
    p0 = path[0]
    pN = path[-1]
    if np.linalg.norm(p0 - pN) > eps:
        path = np.vstack([path, p0])
    return path


def load_deckga_output(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================
# RViz Node
# =========================

class DeckgaMarkers(Node):
    def __init__(
        self,
        frame_id: str,
        topic: str,
        deckga_pkl: str,
        tf: TransformConfig,
        rate_hz: float,
        line_width: float,
        wp_scale: float,
    ) -> None:
        super().__init__("deckga_rviz_paths")

        self.frame_id = frame_id
        self.topic = topic
        self.deckga_pkl = deckga_pkl
        self.tf = tf
        self.line_width = float(line_width)
        self.wp_scale = float(wp_scale)

        self.pub = self.create_publisher(MarkerArray, self.topic, 10)

        data = load_deckga_output(self.deckga_pkl)
        if "deckga_paths" not in data:
            raise KeyError(f"'{self.deckga_pkl}' missing key 'deckga_paths'. Keys: {list(data.keys())}")

        raw_paths = data["deckga_paths"]
        offset_used = data.get("offset_used", None)
        if offset_used is not None:
            offset_used = np.asarray(offset_used, dtype=float).reshape(3,)

        paths = transform_paths(raw_paths=raw_paths, offset_used=offset_used, tf=self.tf)
        self.paths = [enforce_closed_tour(p) for p in paths]

        self.get_logger().info(f"Frame: {self.frame_id}")
        self.get_logger().info(f"Topic: {self.topic}")
        self.get_logger().info(f"DeckGA PKL: {self.deckga_pkl}")
        self.get_logger().info(f"Loaded {len(self.paths)} UAV paths")
        self.get_logger().info(
            f"Transform: coord_mode={self.tf.coord_mode}, scale_xy={self.tf.scale_xy}, "
            f"scale_z={self.tf.scale_z}, z_offset={self.tf.z_offset}, z_min={self.tf.z_min}"
        )

        self.timer = self.create_timer(1.0 / float(rate_hz), self.publish_once)

    def publish_once(self) -> None:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Pylance fix:
        # MarkerArray.markers is typed as a Sequence in stubs, so .append() is not guaranteed.
        # Build a normal Python list, then assign once to ma.markers.
        markers: List[Marker] = []

        palette = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ]

        for i, p in enumerate(self.paths):
            r, g, b = palette[i % len(palette)]

            pts = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in p.tolist()]

            # LINE_STRIP
            m_line = Marker()
            m_line.header.frame_id = self.frame_id
            m_line.header.stamp = now
            m_line.ns = "deckga_route"
            m_line.id = 100 + i
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.scale.x = self.line_width
            m_line.color.a = 1.0
            m_line.color.r = float(r)
            m_line.color.g = float(g)
            m_line.color.b = float(b)
            m_line.pose.orientation.w = 1.0
            m_line.points = pts
            markers.append(m_line)

            # SPHERE_LIST
            m_wp = Marker()
            m_wp.header.frame_id = self.frame_id
            m_wp.header.stamp = now
            m_wp.ns = "deckga_waypoints"
            m_wp.id = 200 + i
            m_wp.type = Marker.SPHERE_LIST
            m_wp.action = Marker.ADD
            m_wp.scale.x = self.wp_scale
            m_wp.scale.y = self.wp_scale
            m_wp.scale.z = self.wp_scale
            m_wp.color.a = 1.0
            m_wp.color.r = float(r)
            m_wp.color.g = float(g)
            m_wp.color.b = float(b)
            m_wp.pose.orientation.w = 1.0
            m_wp.points = pts
            markers.append(m_wp)

        ma.markers = markers
        self.pub.publish(ma)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # You want to run with zero args, so defaults must be correct.
    parser.add_argument("--frame_id", default=DEFAULT_FRAME_ID)
    parser.add_argument("--frame", dest="frame_id", default=argparse.SUPPRESS)  # alias

    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--deckga_pkl", default=DEFAULT_DECKGA_PKL)

    parser.add_argument(
        "--coord_mode",
        default=DEFAULT_COORD_MODE,
        choices=["auto", "shifted", "original"],
    )

    parser.add_argument("--scale", type=float, default=None, help="Legacy: sets BOTH --scale_xy and --scale_z")
    parser.add_argument("--scale_xy", type=float, default=DEFAULT_SCALE_XY)
    parser.add_argument("--scale_z", type=float, default=DEFAULT_SCALE_Z)

    parser.add_argument("--z_offset", type=float, default=DEFAULT_Z_OFFSET)
    parser.add_argument("--z_min", type=float, default=DEFAULT_Z_MIN)

    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--line_width", type=float, default=0.05)
    parser.add_argument("--wp_scale", type=float, default=0.08)

    args = parser.parse_args()

    if args.scale is not None:
        args.scale_xy = float(args.scale)
        args.scale_z = float(args.scale)

    tf = TransformConfig(
        coord_mode=str(args.coord_mode),
        scale_xy=float(args.scale_xy),
        scale_z=float(args.scale_z),
        z_offset=float(args.z_offset),
        z_min=float(args.z_min),
    )

    rclpy.init()
    node = DeckgaMarkers(
        frame_id=str(args.frame_id),
        topic=str(args.topic),
        deckga_pkl=str(args.deckga_pkl),
        tf=tf,
        rate_hz=float(args.rate),
        line_width=float(args.line_width),
        wp_scale=float(args.wp_scale),
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
