#!/usr/bin/env python3
"""
drone_pairwise_distance_logger.py

ROS2 live pairwise distance logger for 3 UAVs.

Purpose:
- During a multi-UAV Gazebo/Aerostack2 flight, subscribe to each drone pose topic.
- Compute pairwise Euclidean distances:
    drone0-drone1
    drone1-drone2
    drone0-drone2
- Save live samples to CSV.
- Save summary CSV with min/max/mean distances.

Recommended Antarctica topics:
    /drone0/ground_truth/pose
    /drone1/ground_truth/pose
    /drone2/ground_truth/pose

Alternative:
    /drone0/self_localization/pose
    /drone1/self_localization/pose
    /drone2/self_localization/pose

Example:
python3 deckga_ros2/drone_pairwise_distance_logger.py \
  --topics /drone0/ground_truth/pose,/drone1/ground_truth/pose,/drone2/ground_truth/pose \
  --msg_type pose_stamped \
  --rate_hz 10 \
  --log_dir results_antarctica_csv \
  --run_tag aspa135_90pts_pairwise_distance_test_run_1
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime
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

        if len(topics) != 3:
            raise ValueError(f"Expected exactly 3 topics, got {len(topics)}: {topics}")
        if len(names) != 3:
            raise ValueError(f"Expected exactly 3 drone names, got {len(names)}: {names}")

        self.topics = topics
        self.names = names
        self.msg_type = msg_type.lower().strip()
        self.rate_hz = float(rate_hz)
        self.print_every_s = float(print_every_s)
        self.stale_after_s = float(stale_after_s)

        ensure_dir(log_dir)
        stamp = now_tag()
        self.samples_csv = os.path.join(log_dir, f"run_{run_tag}_{stamp}_pairwise_distances.csv")
        self.summary_csv = os.path.join(log_dir, f"run_{run_tag}_{stamp}_pairwise_distance_summary.csv")

        self.positions: Dict[str, Optional[np.ndarray]] = {name: None for name in self.names}
        self.last_msg_time: Dict[str, Optional[float]] = {name: None for name in self.names}

        self.t0 = time.perf_counter()
        self.last_print = 0.0

        self.d01_values: List[float] = []
        self.d12_values: List[float] = []
        self.d02_values: List[float] = []
        self.min_values: List[float] = []

        # Use BEST_EFFORT to match typical Gazebo/self-localization sensor QoS.
        # This avoids "incompatible QoS: RELIABILITY" warnings.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.csv_file = open(self.samples_csv, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "time_s",
            f"{self.names[0]}_x", f"{self.names[0]}_y", f"{self.names[0]}_z",
            f"{self.names[1]}_x", f"{self.names[1]}_y", f"{self.names[1]}_z",
            f"{self.names[2]}_x", f"{self.names[2]}_y", f"{self.names[2]}_z",
            f"d_{self.names[0]}_{self.names[1]}_m",
            f"d_{self.names[1]}_{self.names[2]}_m",
            f"d_{self.names[0]}_{self.names[2]}_m",
            "min_pairwise_m",
            "stale_any",
        ])
        self.csv_file.flush()

        # Subscriptions
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

        p0 = self.positions[self.names[0]]
        p1 = self.positions[self.names[1]]
        p2 = self.positions[self.names[2]]

        if p0 is None or p1 is None or p2 is None:
            return

        d01 = dist3(p0, p1)
        d12 = dist3(p1, p2)
        d02 = dist3(p0, p2)
        dmin = min(d01, d12, d02)
        stale = self.stale_any(now)

        self.d01_values.append(d01)
        self.d12_values.append(d12)
        self.d02_values.append(d02)
        self.min_values.append(dmin)

        self.csv_writer.writerow([
            f"{t:.6f}",
            f"{p0[0]:.6f}", f"{p0[1]:.6f}", f"{p0[2]:.6f}",
            f"{p1[0]:.6f}", f"{p1[1]:.6f}", f"{p1[2]:.6f}",
            f"{p2[0]:.6f}", f"{p2[1]:.6f}", f"{p2[2]:.6f}",
            f"{d01:.6f}",
            f"{d12:.6f}",
            f"{d02:.6f}",
            f"{dmin:.6f}",
            int(stale),
        ])

        # Flush often so data is safe even if Ctrl+C happens.
        self.csv_file.flush()

        if now - self.last_print >= self.print_every_s:
            print(
                f"[{t:8.2f}s] "
                f"d01={d01:7.3f} m | "
                f"d12={d12:7.3f} m | "
                f"d02={d02:7.3f} m | "
                f"min={dmin:7.3f} m"
            )
            self.last_print = now

    def write_summary(self) -> None:
        def stats(values: List[float]) -> Tuple[float, float, float]:
            if not values:
                return (math.nan, math.nan, math.nan)
            arr = np.asarray(values, dtype=float)
            return (float(np.min(arr)), float(np.mean(arr)), float(np.max(arr)))

        d01_min, d01_mean, d01_max = stats(self.d01_values)
        d12_min, d12_mean, d12_max = stats(self.d12_values)
        d02_min, d02_mean, d02_max = stats(self.d02_values)
        overall_min, overall_mean, overall_max = stats(self.min_values)

        total_samples = len(self.min_values)
        duration_s = time.perf_counter() - self.t0

        with open(self.summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "samples",
                "duration_s",
                f"min_d_{self.names[0]}_{self.names[1]}_m",
                f"mean_d_{self.names[0]}_{self.names[1]}_m",
                f"max_d_{self.names[0]}_{self.names[1]}_m",
                f"min_d_{self.names[1]}_{self.names[2]}_m",
                f"mean_d_{self.names[1]}_{self.names[2]}_m",
                f"max_d_{self.names[1]}_{self.names[2]}_m",
                f"min_d_{self.names[0]}_{self.names[2]}_m",
                f"mean_d_{self.names[0]}_{self.names[2]}_m",
                f"max_d_{self.names[0]}_{self.names[2]}_m",
                "overall_min_pairwise_m",
                "overall_mean_min_pairwise_m",
                "overall_max_min_pairwise_m",
                "samples_csv",
            ])
            w.writerow([
                total_samples,
                f"{duration_s:.6f}",
                f"{d01_min:.6f}", f"{d01_mean:.6f}", f"{d01_max:.6f}",
                f"{d12_min:.6f}", f"{d12_mean:.6f}", f"{d12_max:.6f}",
                f"{d02_min:.6f}", f"{d02_mean:.6f}", f"{d02_max:.6f}",
                f"{overall_min:.6f}", f"{overall_mean:.6f}", f"{overall_max:.6f}",
                self.samples_csv,
            ])

        print("\n=== Pairwise distance summary ===")
        print(f"Samples: {total_samples}")
        print(f"Duration: {duration_s:.2f} s")
        print(f"{self.names[0]}-{self.names[1]}: min={d01_min:.3f} mean={d01_mean:.3f} max={d01_max:.3f} m")
        print(f"{self.names[1]}-{self.names[2]}: min={d12_min:.3f} mean={d12_mean:.3f} max={d12_max:.3f} m")
        print(f"{self.names[0]}-{self.names[2]}: min={d02_min:.3f} mean={d02_mean:.3f} max={d02_max:.3f} m")
        print(f"Overall minimum pairwise distance: {overall_min:.3f} m")
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
        help="Comma-separated pose topics for drone0,drone1,drone2.",
    )
    parser.add_argument(
        "--names",
        default="drone0,drone1,drone2",
        help="Comma-separated names used in CSV headers.",
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