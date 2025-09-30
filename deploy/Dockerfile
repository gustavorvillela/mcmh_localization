# Full ROS Noetic desktop (with Gazebo, RViz, RQt, etc.)
FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Install Python packages and ROS build tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-rosdep \
    python3-colcon-common-extensions \
    build-essential \
    tmux \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install TurtleBot3 packages
RUN apt-get update && apt-get install -y \
    ros-noetic-turtlebot3 \
    ros-noetic-turtlebot3-simulations \
    && rm -rf /var/lib/apt/lists/*

# Python scientific stack
RUN pip3 install --no-cache-dir --upgrade \
    numpy==1.24.4 \
    scipy \
    numba \
    matplotlib

# Initialize rosdep
RUN rosdep init || true && rosdep update

# Set up catkin workspace
WORKDIR /root/catkin_ws
COPY ./app src/mcmh_localization

# Install package dependencies
RUN rosdep install --from-paths src --ignore-src -r -y

# Build workspace
RUN . /opt/ros/noetic/setup.sh && catkin_make

# Source ROS and workspace automatically
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export TURTLEBOT3_MODEL=waffle" >> /root/.bashrc

CMD ["bash"]
