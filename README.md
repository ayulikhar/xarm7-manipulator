# xArm7 Manipulator

MuJoCo-based manipulation experiments with the **UFACTORY xArm7 7-DOF robotic arm**.

This project explores how task-level manipulation behaviors can be translated into robot motion using **inverse kinematics, Cartesian planning, joint-space control, trajectory generation, contact sensing, and task-specific verification**.

The current implementation includes two manipulation behaviors:

* **Button Pressing**: approach, align, press a physical button, detect contact, and verify the action through an indicator.
* **Peg Insertion**: align a peg with a matching hole, maintain the required orientation, and insert the peg to a defined depth.

The robot model is represented in MJCF and runs in MuJoCo 2.3.3 or later.

---
## Demo:
https://github.com/user-attachments/assets/7f09f27b-d7d4-428d-9e43-e328a5390789

## Manipulation Behaviors

### 1. Button Pressing

The button-press task models a simple contact-rich manipulation problem.

The xArm7 must move from its home configuration toward a button mounted on a panel, align the end effector with the button, press it far enough to actuate the button mechanism, hold the press, and then retract.

The behavior follows:

```text
Home
  ↓
Lift
  ↓
Move above button
  ↓
Descend
  ↓
Press
  ↓
Hold
  ↓
Retract
  ↓
Return Home
```

#### Task setup

The MuJoCo scene contains:

* xArm7 with its end-effector/gripper
* A table and mounting panel
* A button with a **prismatic joint**
* A simulated indicator lamp
* A MuJoCo touch sensor attached to the button

The button has a limited linear travel of 12 mm, allowing the simulation to represent the physical displacement produced by a press.

#### Motion generation

The controller does not directly teleport the arm between joint configurations.

Instead, it generates intermediate Cartesian targets for the TCP and solves inverse kinematics at each step.

For a Cartesian segment:

```text
p(t) = p_start + α(t)(p_goal - p_start)
```

the controller computes a corresponding joint configuration using the robot Jacobian.

The IK solver uses the relationship:

```text
Δx = J(q) Δq
```

and computes joint updates using the Jacobian pseudoinverse.

The implementation uses a separate MuJoCo data structure for IK calculations. This allows the controller to solve candidate configurations without directly disturbing the live simulation state.

The controller also constrains the resulting joint configurations to the robot's joint limits.

#### Why Cartesian interpolation?

For a button press, the direction of approach matters.

The end effector should travel toward the button along a controlled Cartesian path rather than taking an arbitrary joint-space route. The controller therefore interpolates the TCP position between waypoints and repeatedly solves IK from the previous solution.

This produces a sequence of motion phases:

```text
Home → Lift → Approach → Descend → Press
```

followed by:

```text
Press → Retract → Return Home
```

The implementation keeps the gripper state throughout the approach and releases it during the final return-to-home phase.

#### Contact and task verification

The button is not considered pressed simply because the robot reached a target pose.

A MuJoCo touch sensor measures interaction with the button. When the sensor exceeds the configured threshold, the controller marks the button as pressed and changes the simulated lamp state.

The demo therefore verifies three task-level conditions:

```text
Button pressed
Lamp activated during press/hold
Robot returned to home
```

This separates **motion completion** from **task completion**.

### Run

```bash
python3 button_press_demo.py
```

---

## 2. Peg Insertion

The peg-insertion task introduces a more constrained manipulation problem.

Here, a peg is rigidly attached to the xArm7 end effector and must be aligned with a matching hole in a target block before being inserted.

The task is:

```text
Initial Pose
     ↓
Approach Hole
     ↓
Align Peg
     ↓
Enter Hole
     ↓
Insert to Bottom
```

Unlike the button task, successful insertion depends strongly on both **position and orientation**.

### Geometric targets

The scene defines three important task locations:

```text
peg_tip
hole_entry
hole_bottom
```

The controller first positions the peg above the hole, aligns it with the required orientation, moves to the hole entrance, and finally solves for a configuration where the peg reaches the bottom of the hole.

The desired peg orientation points the peg axis downward into the hole.

### Position + orientation IK

The insertion controller solves a 6-DOF end-effector IK problem:

```text
             ┌─ Position error
Task error = │
             └─ Orientation error
```

The Jacobian combines translational and rotational components:

```text
J = [ J_position ]
    [ J_rotation ]
```

and the controller computes joint updates using a **damped least-squares formulation**:

```text
Δq = Jᵀ (J Jᵀ + λI)⁻¹ e
```

This provides a more stable solution near configurations where the Jacobian becomes poorly conditioned.

The controller also limits the magnitude of each joint update and respects the xArm7 joint limits.

### Waypoint-based motion

Once the IK solver determines valid joint configurations for the important task poses, the robot executes them as a sequence of waypoints:

```text
q_start
   ↓
q_approach
   ↓
q_entry
   ↓
q_bottom
```

The motion between waypoints uses a **minimum-jerk trajectory profile**:

```text
s(t) = 10t³ - 15t⁴ + 6t⁵
```

This provides a smooth scalar interpolation between consecutive joint configurations rather than an abrupt transition.

### Insertion verification

The controller evaluates the final distance between the peg tip and the bottom of the hole.

The task reports:

```text
INSERTION SUCCESS
```

when the final distance falls below the configured tolerance.

The script also supports a headless mode for automated task verification:

```bash
python3 insertion_demo.py --headless
```

This makes the behavior useful not only as a visual simulation but also as a repeatable manipulation test.

---

## What I Learned Through These Behaviors

These experiments helped connect the mathematical and implementation sides of robotic manipulation.

### Forward and inverse kinematics

The robot model provides the relationship between joint configurations and end-effector pose.

The behaviors then solve the inverse problem:

```text
Desired TCP pose
      ↓
Inverse Kinematics
      ↓
Joint configuration
      ↓
Robot motion
```

### Jacobians

The Jacobian provides the local relationship between joint velocities/configuration changes and end-effector motion.

The project uses this relationship for both:

* position-oriented button pressing
* position + orientation constrained peg insertion

### Cartesian vs. joint-space planning

The two behaviors highlight why the choice of representation matters.

For the button task, Cartesian interpolation provides controlled motion toward the contact surface.

For insertion, task-space targets are converted into joint-space waypoints and then executed using smooth trajectories.

### Trajectory generation

The project uses interpolation rather than instantaneous changes in joint commands.

The insertion task uses a minimum-jerk profile to produce smoother transitions between manipulation waypoints.

### Contact and task sensing

The button task introduces sensing into the control loop.

The robot does not simply assume that reaching the desired position means the task succeeded. The controller uses the simulated touch sensor to determine whether the button actually received contact.

### Task-level verification

Both behaviors include explicit success criteria.

For example:

```text
Button:
    contact detected
    + lamp activated
    + robot returned home

Peg insertion:
    peg tip reaches insertion depth
    + final distance within tolerance
```

This moves the project from simple robot animation toward **behavior-level manipulation**.

---

## Project Structure

```text
xarm7-manipulator/
│
├── assets/
│   └── Robot meshes and visual assets
│
├── button_press_demo.py
│   └── Button pressing behavior
│
├── button_scene.xml
│   └── Button manipulation environment
│
├── insertion_demo.py
│   └── Peg insertion behavior
│
├── debug_retract_diag.py
│   └── Motion/retraction diagnostics
│
├── scene.xml
│   └── Base MuJoCo environment
│
├── xarm7.xml
│   └── xArm7 MJCF model
│
├── xarm7_nohand.xml
│   └── xArm7 model without hand
│
└── hand.xml
    └── Hand/gripper model
```

The repository also contains the original xArm7 MJCF description and the supporting model assets. The model derives from the publicly available xArm7 URDF and adds MuJoCo-specific actuators and scene configuration.

---

## Getting Started

### Requirements

* Python 3
* MuJoCo 2.3.3+
* NumPy
* MuJoCo Python bindings

Install the Python MuJoCo package:

```bash
pip install mujoco numpy
```

### Button press

```bash
python3 button_press_demo.py
```

### Peg insertion

```bash
python3 insertion_demo.py
```

For headless insertion verification:

```bash
python3 insertion_demo.py --headless
```

---

## Current Scope

The current implementation focuses on **model-based manipulation in simulation**.

The behaviors are deliberately task-specific. The next step is to move from predefined task waypoints toward more general manipulation and planning methods.

Potential directions include:

* Collision-aware motion planning
* A* and sampling-based planners
* Visual servoing
* Object pose estimation
* Force/contact-aware insertion
* ROS 2 integration
* Grasping and pick-and-place
* Learning-based manipulation
* Reinforcement learning

---

## Model

The xArm7 model is represented using MuJoCo's MJCF format.

The original model conversion involved preserving visual geometry, loading the URDF into MuJoCo, restructuring common properties through `<default>` elements, adding actuators, and creating a simulation scene around the robot.

The repository requires **MuJoCo 2.3.3 or later**.

---

## License

See [`LICENSE`](LICENSE) for the applicable license information.
