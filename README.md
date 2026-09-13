# xArm7 Manipulator

MuJoCo-based simulation and control experiments for the **UFACTORY xArm7 7-DOF robotic manipulator**.

The repository currently focuses on manipulation tasks in simulation, including a button-pressing task with Cartesian motion, Jacobian-based inverse kinematics, contact sensing, and task-state verification.

## Overview

This project provides a lightweight MuJoCo environment for experimenting with xArm7 manipulation and control.

The current button-press demo executes the following sequence:

```text
Home
  ↓
Lift
  ↓
Move above button
  ↓
Approach
  ↓
Press
  ↓
Hold
  ↓
Retract
  ↓
Return home
```

The simulation uses a contact sensor to detect the button press. When the button is successfully pressed, the simulated indicator lamp changes state, providing a visual confirmation of task completion.

## Features

* xArm7 MJCF model for MuJoCo
* 7-DOF arm control
* Position and orientation-aware Jacobian IK
* Cartesian end-effector interpolation
* Smooth joint-space control
* Simulated button with physical travel
* Contact-based button detection
* Visual lamp feedback
* Automatic return-to-home verification
* Additional manipulation experiments

## Button Press Demo

`button_press_demo.py` controls the xArm7 in `button_scene.xml`.

The controller uses a damped least-squares style Jacobian IK solver to generate joint targets while preserving the end-effector orientation. Cartesian interpolation keeps the end-effector trajectory close to the intended straight-line path during approach and retraction.

The task controller also monitors a MuJoCo touch sensor attached to the button. A successful press activates the simulated lamp.

At the end of each cycle, the controller reports:

```text
Button pressed: True
Lamp turned on during press/hold: True
Returned to home: True
```

### Run

Install MuJoCo and Python dependencies, then run:

```bash
python button_press_demo.py
```

The simulation opens in the MuJoCo viewer and continuously executes the button-press cycle.

## Repository Structure

```text
xarm7-manipulator/
│
├── assets/
│   └── Robot meshes and visual assets
│
├── button_press_demo.py
│   └── Button-press task controller
│
├── button_scene.xml
│   └── MuJoCo scene containing the xArm7 and button
│
├── debug_retract_diag.py
│   └── Diagnostics for motion and retraction behavior
│
├── insertion_demo.py
│   └── Manipulation/insertion experiment
│
├── hand.xml
│   └── xArm7 hand/gripper model
│
├── scene.xml
│   └── Base MuJoCo scene
│
├── xarm7.xml
│   └── xArm7 MJCF model
│
└── xarm7_nohand.xml
    └── xArm7 model without the hand
```

## Technical Approach

### Inverse Kinematics

The button-press controller computes the desired end-effector position and solves for the corresponding arm joint configuration using the MuJoCo site Jacobian.

The solver uses the current joint configuration as a warm start and constrains the solution within the robot's joint limits.

### Cartesian Motion

Rather than directly interpolating between arbitrary joint configurations, the controller interpolates the desired TCP position and solves IK along the Cartesian trajectory.

This produces controlled motion for:

* lifting from the home pose
* approaching the button
* descending onto the button
* pressing
* retracting
* returning to the home position

### Contact Detection

A MuJoCo touch sensor detects interaction between the end effector and the button.

The sensor state drives the simulated indicator lamp, making it possible to verify that the robot physically reaches the button rather than simply reaching a predefined joint configuration.

## Model

The xArm7 model is represented in **MJCF** and includes the arm, actuators, and hand/gripper components.

The model is derived from the publicly available xArm7 URDF description and adapted for MuJoCo simulation.

## Requirements

* Python 3
* MuJoCo
* NumPy
* MuJoCo Python bindings

The model currently targets **MuJoCo 2.3.3 or later**.

## Future Work

Planned experiments include:

* Motion planning with A*
* Collision-aware manipulation
* Improved end-effector control
* Object interaction and insertion tasks
* Grasping and pick-and-place
* Vision-guided manipulation
* ROS 2 integration
* Reinforcement learning for manipulation

## Motivation

This project serves as a sandbox for studying robotic manipulation, motion planning, and control through progressively more complex simulation tasks.

The long-term goal is to move from scripted manipulation toward planners and learning-based controllers that can generalize across different tasks and environments.

## License

See [`LICENSE`](LICENSE) for licensing information.
