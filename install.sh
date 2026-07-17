#!/bin/bash
set -e

# Set ROS workspace and source directory
cd ..


# Clone TurtleBot3 repositories from source
git clone -b noetic git@github.com:gustavorvillela/turtlebot3_diff.git
git clone https://github.com/alspitz/cpu_monitor.git


echo "TurtleBot3 packages cloned from source successfully."

