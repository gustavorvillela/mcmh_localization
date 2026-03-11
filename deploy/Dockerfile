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
    ca-certificates \
    git \
    xauth \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Python scientific stack
RUN pip3 install --no-cache-dir --upgrade \
    numpy==1.24.4 \
    scipy \
    numba \
    matplotlib

# Install default ROS packages for TurtleBot3
RUN apt-get update && apt-get install -y \
    ros-noetic-turtlebot3-simulations \
    ros-noetic-turtlebot3-msgs \
    ros-noetic-navigation \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
COPY /mcmh_localization/local_rosdep/sources.list.d/ /tmp/sources.list.d

RUN mkdir -p /etc/ros/rosdep/sources.list.d && \
    cp /tmp/sources.list.d/20-default.list /etc/ros/rosdep/sources.list.d/

ENV ROSDISTRO_INDEX_URL=https://cdn.jsdelivr.net/gh/ros/rosdistro@master/index-v4.yaml
RUN rosdep update

# Set up catkin workspace
WORKDIR /root/catkin_ws
COPY mcmh_localization/app src/mcmh_localization

COPY ./turtlebot3_diff src/turtlebot3



# Install package dependencies
RUN rosdep install --from-paths src --ignore-src -r -y

RUN rm -rf ./src/turtlebot3

# Build workspace
RUN . /opt/ros/noetic/setup.sh && catkin_make

# Source ROS and workspace automatically
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export TURTLEBOT3_MODEL=waffle" >> /root/.bashrc && \
    echo "export DISABLE_ROS1_EOL_WARNINGS=1" >> /root/.bashrc

RUN sudo ln -s /usr/bin/python3 /usr/bin/python


CMD ["bash"]
