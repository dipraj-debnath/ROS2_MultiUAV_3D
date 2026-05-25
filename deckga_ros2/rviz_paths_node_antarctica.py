#!/usr/bin/env python3
"""
RViz visualization for DECK-GA paths in Antarctica world.

Matches deckga_execute_antarctica.py transform:
- auto/original/shifted
- unshift by offset_used if needed
- XY scale + center
- optional XY clamp
- Z mapping: zin->[zout] + z_base

Publishes MarkerArray:
- LINE_STRIP per UAV (thicker width)
- SPHERE markers for start points
"""

import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


DEFAULT_FRAME_ID = "earth"


def load_deckga_pkl(pkl_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if "deckga_paths" not in data:
        raise KeyError(f"'deckga_paths' missing. keys={list(data.keys())}")
    paths = [np.array(p, dtype=float) for p in data["deckga_paths"]]
    offset = np.array(data.get("offset_used", [0.0, 0.0, 0.0]), dtype=float).reshape(3,)
    return paths, offset


def decide_coord_mode(paths: List[np.ndarray], offset_used: np.ndarray, coord_mode: str) -> str:
    coord_mode = coord_mode.lower().strip()
    if coord_mode in ("original", "shifted"):
        return coord_mode
    if coord_mode != "auto":
        raise ValueError("coord_mode must be one of: auto, original, shifted")

    if np.linalg.norm(offset_used[:2]) > 1e-9:
        all_min_xy = min(float(np.min(p[:, :2])) for p in paths) if paths else 0.0
        if all_min_xy >= -1e-6:
            return "shifted"
    return "original"


def map_range(v: np.ndarray, in_min: float, in_max: float, out_min: float, out_max: float) -> np.ndarray:
    v = np.clip(v, in_min, in_max)
    if abs(in_max - in_min) < 1e-9:
        return np.full_like(v, out_min)
    t = (v - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def transform_paths(
    paths: List[np.ndarray],
    offset_used: np.ndarray,
    coord_mode: str,
    xy_scale: float,
    xy_center_x: float,
    xy_center_y: float,
    clamp_xy: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    zin_min: float,
    zin_max: float,
    zout_min: float,
    zout_max: float,
    z_base: float,
) -> List[np.ndarray]:
    mode = decide_coord_mode(paths, offset_used, coord_mode)
    out: List[np.ndarray] = []

    for p in paths:
        q = p.copy()

        if mode == "shifted":
            q[:, 0] -= offset_used[0]
            q[:, 1] -= offset_used[1]
            q[:, 2] -= offset_used[2]

        q[:, 0] = q[:, 0] * float(xy_scale) + float(xy_center_x)
        q[:, 1] = q[:, 1] * float(xy_scale) + float(xy_center_y)

        if bool(clamp_xy):
            q[:, 0] = np.clip(q[:, 0], float(x_min), float(x_max))
            q[:, 1] = np.clip(q[:, 1], float(y_min), float(y_max))

        rel_z = map_range(q[:, 2], float(zin_min), float(zin_max), float(zout_min), float(zout_max))
        q[:, 2] = rel_z + float(z_base)

        out.append(q)

    return out


def color_for(i: int):
    """
    Distinct path colours for multi-UAV experiments.
    Supports 2, 3, 4, 5+ UAVs without repeating red/green/blue immediately.
    Returns (r, g, b, a).
    """
    palette = [
        (1.0, 0.0, 0.0, 1.0),   # UAV0 red
        (0.0, 0.8, 0.0, 1.0),   # UAV1 green
        (0.0, 0.2, 1.0, 1.0),   # UAV2 blue
        (1.0, 0.5, 0.0, 1.0),   # UAV3 orange
        (0.7, 0.0, 1.0, 1.0),   # UAV4 purple
        (0.0, 1.0, 1.0, 1.0),   # UAV5 cyan
        (1.0, 0.0, 1.0, 1.0),   # UAV6 magenta
        (1.0, 1.0, 0.0, 1.0),   # UAV7 yellow
    ]
    return palette[i % len(palette)]


class DeckgaRvizAntarctica(Node):
    def __init__(self, args):
        super().__init__("deckga_rviz_paths_antarctica")

        self.frame_id = args.frame_id
        self.pub = self.create_publisher(MarkerArray, args.topic, 10)

        raw_paths, offset_used = load_deckga_pkl(args.deckga_pkl)
        raw_paths = raw_paths[: args.num_uavs]

        self.paths_m = transform_paths(
            raw_paths,
            offset_used=offset_used,
            coord_mode=args.coord_mode,
            xy_scale=args.xy_scale,
            xy_center_x=args.xy_center_x,
            xy_center_y=args.xy_center_y,
            clamp_xy=args.clamp_xy,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            zin_min=args.zin_min,
            zin_max=args.zin_max,
            zout_min=args.zout_min,
            zout_max=args.zout_max,
            z_base=args.z_base,
        )

        self.line_width = float(args.line_width)
        self.wp_scale = float(args.wp_scale)
        self.rate_hz = float(args.rate_hz)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

    def on_timer(self):
        ma = MarkerArray()

        now = self.get_clock().now().to_msg()

        # paths
        for i, p in enumerate(self.paths_m):
            if len(p) == 0:
                continue

            line = Marker()
            line.header.frame_id = self.frame_id
            line.header.stamp = now
            line.ns = f"deckga_path_uav_{i}"
            line.id = 100 + i
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = self.line_width  # THICKER => less “blurry”
            r, g, b, a = color_for(i)
            line.color.r, line.color.g, line.color.b, line.color.a = r, g, b, a

            for row in p:
                pt = Point()
                pt.x, pt.y, pt.z = map(float, row)
                line.points.append(pt)

            ma.markers.append(line)

            # start sphere
            start = Marker()
            start.header.frame_id = self.frame_id
            start.header.stamp = now
            start.ns = f"deckga_start_uav_{i}"
            start.id = 200 + i
            start.type = Marker.SPHERE
            start.action = Marker.ADD
            start.scale.x = self.wp_scale
            start.scale.y = self.wp_scale
            start.scale.z = self.wp_scale
            start.color.r, start.color.g, start.color.b, start.color.a = r, g, b, 1.0
            start.pose.position.x = float(p[0, 0])
            start.pose.position.y = float(p[0, 1])
            start.pose.position.z = float(p[0, 2])
            ma.markers.append(start)

        self.pub.publish(ma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deckga_pkl", default="deckga_ros2/data/deckga_output_antarctica.pkl")
    ap.add_argument("--num_uavs", type=int, default=3)
    ap.add_argument("--frame_id", default=DEFAULT_FRAME_ID)
    ap.add_argument("--topic", default="/deckga/paths_antarctica")

    ap.add_argument("--coord_mode", default="auto", choices=["auto", "original", "shifted"])

    ap.add_argument("--xy_scale", type=float, default=0.30)
    ap.add_argument("--xy_center_x", type=float, default=6.0)
    ap.add_argument("--xy_center_y", type=float, default=4.5)

    ap.add_argument("--clamp_xy", action="store_true")
    ap.add_argument("--x_min", type=float, default=-24.0)
    ap.add_argument("--x_max", type=float, default=36.0)
    ap.add_argument("--y_min", type=float, default=-25.0)
    ap.add_argument("--y_max", type=float, default=35.0)

    ap.add_argument("--zin_min", type=float, default=10.0)
    ap.add_argument("--zin_max", type=float, default=100.0)
    ap.add_argument("--zout_min", type=float, default=5.0)
    ap.add_argument("--zout_max", type=float, default=15.0)
    ap.add_argument("--z_base", type=float, default=32.2)

    # visualization thickness
    ap.add_argument("--line_width", type=float, default=0.18)
    ap.add_argument("--wp_scale", type=float, default=0.35)

    ap.add_argument("--rate_hz", type=float, default=2.0)

    args = ap.parse_args()

    rclpy.init()
    node = None
    try:
        node = DeckgaRvizAntarctica(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
