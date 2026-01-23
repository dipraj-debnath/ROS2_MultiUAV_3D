#!/usr/bin/env bash
set -e

# Run from project_gazebo folder
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Gazebo Sim (Harmonic) resource path
export GZ_SIM_RESOURCE_PATH="${THIS_DIR}/models:${THIS_DIR}/worlds${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"

# Backwards-compatible name (some tooling still checks this)
export IGN_GAZEBO_RESOURCE_PATH="${THIS_DIR}/models:${THIS_DIR}/worlds${IGN_GAZEBO_RESOURCE_PATH:+:${IGN_GAZEBO_RESOURCE_PATH}}"

echo "GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
echo "IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH"
