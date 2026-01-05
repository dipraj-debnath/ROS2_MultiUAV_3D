## 🧩 Overview
This lesson demonstrates how to set up and run **Aerostack2** examples using **Gazebo** and **RViz2** for both single-drone and multi-drone missions.

You will:
- Launch Aerostack2 with Gazebo  
- Visualize in RViz2  
- Fly a single UAV (autonomous square mission)  
- Fly multiple UAVs (swarm mission)

---

## 🛠️ Prerequisites

Before starting:
```bash
sudo apt install -y tmux ruby-full build-essential
sudo gem install tmuxinator
sudo apt install -y ros-humble-aerostack2 ros-humble-as2-keyboard-teleoperation
Then source your ROS 2 and Aerostack 2 environments:

bash
Copy code
source /opt/ros/humble/setup.bash
source /opt/ros/humble/share/aerostack2/local_setup.bash
📂 Folder Used
All work is done inside:

bash
Copy code
~/Documents/GitHub/ros2_python_learning/day8_aerostack2/
✈️ Part A — Single UAV (Gazebo Example)
1️⃣ Launch the simulation
bash
Copy code
cd ~/Documents/GitHub/ros2_python_learning/day8_aerostack2/02_examples_gazebo_project/project_gazebo
./launch_as2.bash
This opens:

Gazebo (world + drone0)

Aerostack2 stack

Alphanumeric telemetry terminal

Wait ~15 s for everything to initialize.

2️⃣ Launch the ground station
Open a new terminal:

bash
Copy code
source /opt/ros/humble/setup.bash
source /opt/ros/humble/share/aerostack2/local_setup.bash
cd ~/Documents/GitHub/ros2_python_learning/day8_aerostack2/02_examples_gazebo_project/project_gazebo
./launch_ground_station.bash -t -v
This opens:

RViz2 visualization

Teleoperation GUI (Takeoff / Land etc.)

3️⃣ Run the mission
Open another terminal:

bash
Copy code
source /opt/ros/humble/setup.bash
source /opt/ros/humble/share/aerostack2/local_setup.bash
cd ~/Documents/GitHub/ros2_python_learning/day8_aerostack2/02_examples_gazebo_project/project_gazebo
python3 mission.py
✅ The drone will take off, fly a square path, and land.

4️⃣ Stop everything
bash
Copy code
./stop.bash
🐝 Part B — Multi-UAV (Swarm Mission)
1️⃣ Launch multi-UAV Aerostack2 + Gazebo
bash
Copy code
source /opt/ros/humble/setup.bash
source /opt/ros/humble/share/aerostack2/local_setup.bash
cd ~/Documents/GitHub/ros2_python_learning/day8_aerostack2/02_examples_gazebo_project/project_gazebo
./launch_as2.bash -m
This will spawn drone0, drone1, and drone2.

If Gazebo appears black, restart with software rendering:

bash
Copy code
export LIBGL_ALWAYS_SOFTWARE=1
./launch_as2.bash -m
2️⃣ Verify connection (optional)
bash
Copy code
ros2 topic echo /drone0/platform/status
ros2 topic echo /drone1/platform/status
ros2 topic echo /drone2/platform/status
Expect each to show connected: true.

3️⃣ Launch the ground station (multi-UAV)
bash
Copy code
source /opt/ros/humble/setup.bash
source /opt/ros/humble/share/aerostack2/local_setup.bash
cd ~/Documents/GitHub/ros2_python_learning/day8_aerostack2/02_examples_gazebo_project/project_gazebo
./launch_ground_station.bash -m -t -v
You will see RViz + multi-drone Teleop GUI.

4️⃣ Run the swarm mission
bash
Copy code
source /opt/ros/humble/setup.bash
source /opt/ros/humble/share/aerostack2/local_setup.bash
cd ~/Documents/GitHub/ros2_python_learning/day8_aerostack2/02_examples_gazebo_project/project_gazebo
python3 mission_swarm.py
Respond to prompts:

bash
Copy code
Takeoff? (y/n): y
Go to? (y/n): y
Replay? (y/n): n
Land? (y/n): y
✅ All drones take off, perform their paths, and land.

5️⃣ Stop everything
bash
Copy code
./stop.bash
pkill -f gzserver || true
pkill -f gzclient || true
🧠 Tips
Always source both setup files in every terminal.

Wait for [droneX.adapter]: Mission Interpreter Adapter ready before running missions.

Use rviz to visualize positions and as2_alphanumeric_viewer for telemetry.

If the teleop GUI doesn’t open, install:

bash
Copy code
pip install --user PySimpleGUI-4-foss
🎥 Verification
✅ Single UAV mission — drone takes off, flies square, lands
✅ Multi-UAV mission — all three drones take off, fly coordinated pattern, land
✅ RViz and Gazebo visualize live drone motion

