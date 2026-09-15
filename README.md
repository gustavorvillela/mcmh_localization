# 🧠 MCMH Localization — Setup Guide

This project provides a **Metropolis–Hastings–based localization system** integrated with the **TurtleBot3 simulation** in ROS Noetic.  
Follow the steps below for your first setup and containerized deployment.

---

## 🚀 1. Clone the Repository

Clone this repository directly into your `catkin_ws/src` folder:

```bash
cd ~/catkin_ws/src
git clone <link_to_this_repo>.git
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
cd ~/catkin_ws
catkin_make
source devel/setup.bash
roslaunch mcmh_localization <your_launch_file>.launch
```
## 5. Single-run simulations 

### 5.1. Running MH-MCL/3CML simulation

To run online simulation and estimation using the MH estimators, set the params in the mcmh_localization.launch file or on terminal and launch:

```bash
roslaunch mcmh_localization mcmh_localization.launch 
```

### 5.2. Creating map
Create a new map for estimation by running (choose params in same launch file/on terminal):

```bash
roslaunch mcmh_localization create_map.launch 
```

### 5.3. Recording scenario
Create a new scenario bag for estimation by running (choose params in same launch file/on terminal):

```bash
roslaunch mcmh_localization record_sim.launch 
```

### 5.4. Replaying scenario
To replay an existing scenario bag for estimation run (choose params in same launch file/on terminal):

```bash
roslaunch mcmh_localization play_bag.launch 
```

## 6. Batch simulations

### 6.1. Internal Sweep

To run repeated internal runs of 3MCL for parameter search (choose params in same launch file/on terminal):

```bash
roscd mcmh_localization/scripts
./run_internal_sweep.sh 
```

### 6.2. Full  Sweep

To run repeated runs over all Particle Filters (choose params in same launch file/on terminal):

```bash
roscd mcmh_localization/scripts
./run_particle_sweep.sh <desired_bag_file>.bag
```

