#!/usr/bin/env python3
"""
drone_pairwise_distance_logger.py

ROS2 live pairwise distance logger for any number of UAVs.

Purpose:
- During a multi-UAV Gazebo/Aerostack2 flight, subscribe to each drone pose topic.
- Compute all pairwise Euclidean distances automatically.
  Example:
    2 drones -> 1 pair
    3 drones -> 3 pairs
    4 drones -> 6 pairs
    5 drones -> 10 pairs
- Save live samples to CSV.
- Save summary CSV with min/mean/max distance for every pair.

Recommended Antarctica topics:
    /drone0/ground_truth/pose
    /drone1/ground_truth/pose
    /drone2/ground_truth/pose
    ...

Alternative:
    /drone0/self_localization/pose
    /drone1/self_localization/pose
    /drone2/self_localization/pose
    ...

Example for 3 drones:
python3 deckga_ros2/drone_pairwise_distance_logger.py \
  --topics /drone0/ground_truth/pose,/drone1/ground_truth/pose,/drone2/ground_truth/pose \
  --names drone0,drone1,drone2 \
  --msg_type pose_stamped \
  --rate_hz 10 \
  --log_dir results_antarctica_csv \
  --run_tag aspa135_90pts_uav3_pairwise_distance_test_run_1

Example for 5 drones:
python3 deckga_ros2/drone_pairwise_distance_logger.py \
  --topics /drone0/ground_truth/pose,/drone1/ground_truth/pose,/drone2/ground_truth/pose,/drone3/ground_truth/pose,/drone4/ground_truth/pose \
  --names drone0,drone1,drone2,drone3,drone4 \
  --msg_type pose_stamped \
  --rate_hz 10 \
  --log_dir results_antarctica_csv \
  --run_tag aspa135_90pts_uav5_pairwise_distance_test_run_1
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dist3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def pair_label(a: str, b: str) -> str:
    return f"{a}_{b}"


class DronePairwiseDistanceLogger(Node):
    def __init__(
        self,
        topics: List[str],
        names: List[str],
        msg_type: str,
        rate_hz: float,
        log_dir: str,
        run_tag: str,
        print_every_s: float,
        stale_after_s: float,
    ):
        super().__init__("drone_pairwise_distance_logger")

        if len(topics) != len(names):
            raise ValueError(
                f"Number of topics must match number of names. "
                f"Got {len(topics)} topics and {len(names)} names."
            )

        if len(topics) < 2:
            raise ValueError("At least 2 drones/topics are required for pairwise distance logging.")

        self.topics = topics
        self.names = names
        self.msg_type = msg_type.lower().strip()
        self.rate_hz = float(rate_hz)
        self.print_every_s = float(print_every_s)
        self.stale_after_s = float(stale_after_s)

        self.pairs: List[Tuple[str, str]] = list(combinations(self.names, 2))
        self.pair_labels: List[str] = [pair_label(a, b) for a, b in self.pairs]

        ensure_dir(log_dir)
        stamp = now_tag()
        self.samples_csv = os.path.join(log_dir, f"run_{run_tag}_{stamp}_pairwise_distances.csv")
        self.summary_csv = os.path.join(log_dir, f"run_{run_tag}_{stamp}_pairwise_distance_summary.csv")

        self.positions: Dict[str, Optional[np.ndarray]] = {name: None for name in self.names}
        self.last_msg_time: Dict[str, Optional[float]] = {name: None for name in self.names}

        self.t0 = time.perf_counter()
        self.last_print = 0.0

        self.pair_values: Dict[str, List[float]] = {label: [] for label in self.pair_labels}
        self.min_pairwise_values: List[float] = []
        self.closest_pair_values: List[str] = []

        self.global_min_distance: float = math.inf
        self.global_min_pair: str = ""
        self.global_min_time_s: float = 0.0

        # BEST_EFFORT usually matches Gazebo/self-localization sensor QoS.
        # This avoids "incompatible QoS: RELIABILITY" warnings.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.csv_file = open(self.samples_csv, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        header = ["time_s"]

        for name in self.names:
            header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

        for label in self.pair_labels:
            header.append(f"d_{label}_m")

        header.extend([
            "min_pairwise_m",
            "closest_pair",
            "stale_any",
        ])

        self.csv_writer.writerow(header)
        self.csv_file.flush()

        for idx, topic in enumerate(self.topics):
            name = self.names[idx]

            if self.msg_type == "pose_stamped":
                self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, n=name: self.pose_stamped_callback(msg, n),
                    qos,
                )
            elif self.msg_type == "pose":
                self.create_subscription(
                    Pose,
                    topic,
                    lambda msg, n=name: self.pose_callback(msg, n),
                    qos,
                )
            elif self.msg_type == "odometry":
                self.create_subscription(
                    Odometry,
                    topic,
                    lambda msg, n=name: self.odom_callback(msg, n),
                    qos,
                )
            else:
                raise ValueError("--msg_type must be one of: pose_stamped, pose, odometry")

            self.get_logger().info(f"Subscribed {name}: {topic} as {self.msg_type}")

        period = 1.0 / max(self.rate_hz, 1e-6)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(f"Number of drones: {len(self.names)}")
        self.get_logger().info(f"Number of pairwise distances: {len(self.pairs)}")
        self.get_logger().info(f"Pairs: {self.pair_labels}")
        self.get_logger().info(f"Logging live pairwise distances to: {self.samples_csv}")
        self.get_logger().info(f"Summary will be written to: {self.summary_csv}")

    def pose_stamped_callback(self, msg: PoseStamped, name: str) -> None:
        p = msg.pose.position
        self.positions[name] = np.array([float(p.x), float(p.y), float(p.z)], dtype=float)
        self.last_msg_time[name] = time.perf_counter()

    def pose_callback(self, msg: Pose, name: str) -> None:
        p = msg.position
        self.positions[name] = np.array([float(p.x), float(p.y), float(p.z)], dtype=float)
        self.last_msg_time[name] = time.perf_counter()

    def odom_callback(self, msg: Odometry, name: str) -> None:
        p = msg.pose.pose.position
        self.positions[name] = np.array([float(p.x), float(p.y), float(p.z)], dtype=float)
        self.last_msg_time[name] = time.perf_counter()

    def all_positions_available(self) -> bool:
        return all(self.positions[name] is not None for name in self.names)

    def stale_any(self, now: float) -> bool:
        for name in self.names:
            t = self.last_msg_time[name]
            if t is None:
                return True
            if (now - t) > self.stale_after_s:
                return True
        return False

    def timer_callback(self) -> None:
        now = time.perf_counter()
        t = now - self.t0

        if not self.all_positions_available():
            if now - self.last_print >= self.print_every_s:
                missing = [name for name in self.names if self.positions[name] is None]
                self.get_logger().warn(f"Waiting for pose messages from: {missing}")
                self.last_print = now
            return

        stale = self.stale_any(now)

        row = [f"{t:.6f}"]

        for name in self.names:
            p = self.positions[name]
            if p is None:
                row.extend(["nan", "nan", "nan"])
            else:
                row.extend([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])

        pair_distances: Dict[str, float] = {}

        for (a, b), label in zip(self.pairs, self.pair_labels):
            pa = self.positions[a]
            pb = self.positions[b]

            if pa is None or pb is None:
                d = math.nan
            else:
                d = dist3(pa, pb)

            pair_distances[label] = d
            self.pair_values[label].append(d)
            row.append(f"{d:.6f}" if not math.isnan(d) else "nan")

        valid_distances = {
            label: d for label, d in pair_distances.items()
            if not math.isnan(d)
        }

        if valid_distances:
            closest_pair = min(valid_distances, key=valid_distances.get)
            min_pairwise = float(valid_distances[closest_pair])
        else:
            closest_pair = ""
            min_pairwise = math.nan

        self.min_pairwise_values.append(min_pairwise)
        self.closest_pair_values.append(closest_pair)

        if not math.isnan(min_pairwise) and min_pairwise < self.global_min_distance:
            self.global_min_distance = min_pairwise
            self.global_min_pair = closest_pair
            self.global_min_time_s = t

        row.extend([
            f"{min_pairwise:.6f}" if not math.isnan(min_pairwise) else "nan",
            closest_pair,
            int(stale),
        ])

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        if now - self.last_print >= self.print_every_s:
            distance_text = " | ".join([
                f"{label}={pair_distances[label]:.3f} m"
                for label in self.pair_labels
                if not math.isnan(pair_distances[label])
            ])

            print(
                f"[{t:8.2f}s] "
                f"{distance_text} | "
                f"min={min_pairwise:.3f} m ({closest_pair})"
            )
            self.last_print = now

    def _stats(self, values: List[float]) -> Tuple[float, float, float]:
        clean = [v for v in values if not math.isnan(v)]
        if not clean:
            return (math.nan, math.nan, math.nan)
        arr = np.asarray(clean, dtype=float)
        return (float(np.min(arr)), float(np.mean(arr)), float(np.max(arr)))

    def write_summary(self) -> None:
        total_samples = len(self.min_pairwise_values)
        duration_s = time.perf_counter() - self.t0

        summary_header = [
            "num_drones",
            "num_pairs",
            "samples",
            "duration_s",
        ]

        summary_row = [
            int(len(self.names)),
            int(len(self.pairs)),
            int(total_samples),
            f"{duration_s:.6f}",
        ]

        for label in self.pair_labels:
            d_min, d_mean, d_max = self._stats(self.pair_values[label])
            summary_header.extend([
                f"min_d_{label}_m",
                f"mean_d_{label}_m",
                f"max_d_{label}_m",
            ])
            summary_row.extend([
                f"{d_min:.6f}" if not math.isnan(d_min) else "nan",
                f"{d_mean:.6f}" if not math.isnan(d_mean) else "nan",
                f"{d_max:.6f}" if not math.isnan(d_max) else "nan",
            ])

        overall_min, overall_mean, overall_max = self._stats(self.min_pairwise_values)

        summary_header.extend([
            "overall_min_pairwise_m",
            "overall_mean_min_pairwise_m",
            "overall_max_min_pairwise_m",
            "global_min_pair",
            "global_min_time_s",
            "samples_csv",
        ])

        summary_row.extend([
            f"{overall_min:.6f}" if not math.isnan(overall_min) else "nan",
            f"{overall_mean:.6f}" if not math.isnan(overall_mean) else "nan",
            f"{overall_max:.6f}" if not math.isnan(overall_max) else "nan",
            self.global_min_pair,
            f"{self.global_min_time_s:.6f}",
            self.samples_csv,
        ])

        with open(self.summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(summary_header)
            w.writerow(summary_row)

        print("\n=== Pairwise distance summary ===")
        print(f"Number of drones: {len(self.names)}")
        print(f"Number of pairs : {len(self.pairs)}")
        print(f"Samples         : {total_samples}")
        print(f"Duration        : {duration_s:.2f} s")

        for label in self.pair_labels:
            d_min, d_mean, d_max = self._stats(self.pair_values[label])
            print(f"{label}: min={d_min:.3f} mean={d_mean:.3f} max={d_max:.3f} m")

        print(f"Overall minimum pairwise distance: {overall_min:.3f} m")
        print(f"Closest pair at global minimum   : {self.global_min_pair}")
        print(f"Time of global minimum           : {self.global_min_time_s:.3f} s")
        print(f"[LOG] Samples CSV: {self.samples_csv}")
        print(f"[LOG] Summary CSV: {self.summary_csv}")

    def close_files(self) -> None:
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--topics",
        default="/drone0/ground_truth/pose,/drone1/ground_truth/pose,/drone2/ground_truth/pose",
        help="Comma-separated pose topics. Must match --names length.",
    )
    parser.add_argument(
        "--names",
        default="drone0,drone1,drone2",
        help="Comma-separated drone names used in CSV headers.",
    )
    parser.add_argument(
        "--msg_type",
        default="pose_stamped",
        choices=["pose_stamped", "pose", "odometry"],
        help="Message type of input topics.",
    )
    parser.add_argument("--rate_hz", type=float, default=10.0)
    parser.add_argument("--print_every_s", type=float, default=1.0)
    parser.add_argument("--stale_after_s", type=float, default=1.0)
    parser.add_argument("--log_dir", default="results_antarctica_csv")
    parser.add_argument("--run_tag", default="pairwise_distance")

    args = parser.parse_args()

    topics = [x.strip() for x in args.topics.split(",") if x.strip()]
    names = [x.strip() for x in args.names.split(",") if x.strip()]

    rclpy.init()

    node = DronePairwiseDistanceLogger(
        topics=topics,
        names=names,
        msg_type=args.msg_type,
        rate_hz=args.rate_hz,
        log_dir=args.log_dir,
        run_tag=args.run_tag,
        print_every_s=args.print_every_s,
        stale_after_s=args.stale_after_s,
    )

    try:
        print("\nPairwise distance logger running.")
        print("Start your flight executor in another terminal.")
        print("Press Ctrl+C here after the mission finishes.\n")
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nStopping pairwise distance logger...")

    finally:
        node.write_summary()
        node.close_files()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()