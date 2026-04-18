echo "Starting ROS2-Gazebo bridges..."

UAV_NAMES=(uav1 uav2 uav3 uav4)
BRIDGE_PIDS=()

for uav in "${UAV_NAMES[@]}"; do
  echo "Starting ${uav}/cmd_vel bridge..."
  ros2 run ros_gz_bridge parameter_bridge /${uav}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist &
  BRIDGE_PIDS+=("$!")

  sleep 0.5

  echo "Starting ${uav}/odom bridge..."
  ros2 run ros_gz_bridge parameter_bridge /model/${uav}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry --ros-args -r /model/${uav}/odometry:=/${uav}/odom &
  BRIDGE_PIDS+=("$!")

  sleep 0.5

  echo "Starting ${uav}/lidar bridge..."
  ros2 run ros_gz_bridge parameter_bridge /${uav}/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan &
  BRIDGE_PIDS+=("$!")

  sleep 0.5
  echo ""
done

echo "All bridges started!"
echo "Bridge PIDs: ${BRIDGE_PIDS[*]}"
echo ""

trap "echo 'Stopping bridges...'; kill ${BRIDGE_PIDS[*]} 2>/dev/null; exit" SIGINT SIGTERM

wait
