# 🧠 MCMH Localization — Setup Guide

This project provides a **Metropolis–Hastings–based localization system** integrated with the **TurtleBot3 simulation** in ROS Noetic.  
Follow the steps below for your first setup and containerized deployment.

---

## 🚀 1. Clone the Repository

Clone this repository directly into your `catkin_ws/src` folder:

```bash
cd ~/catkin_ws/src
git clone https://github.com/gustavorvillela/mcmh_localization.git
```

---

## ⚙️ 2. Run the Installation Script

Run the installation script to automatically clone and configure the required TurtleBot3 packages (from your custom forks):

```bash
cd ~/catkin_ws/src/mcmh_localization
bash install.sh
```

This step installs dependencies and prepares the workspace for building inside Docker.

## 🐳 3. Build and Launch the Docker Environment

Navigate to the deploy folder and build the Docker container:

```bash
cd ~/catkin_ws/src/mcmh_localization/deploy
docker compose build
```


Then start the container in detached mode:

```bash
docker compose up -d
```

## 💻 4. Access the ROS Noetic Container

Enter the running container interactively:

```bash
docker exec -it ros_noetic_dev bash
```


Inside the container, you can now build and launch your workspace as usual:

```bash
catkin_make
source devel/setup.bash
roslaunch mcmh_localization <your_launch_file>.launch
```
