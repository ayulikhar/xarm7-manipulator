"""Simple, reliable xarm7 button-press demo.

Motion sequence: home -> approach -> press button -> hold (lamp lights) -> retract -> home.
Uses a lightweight Jacobian IK (position-only) computed on a scratch MjData copy so the
live simulation is never perturbed by direct qpos writes.
"""
import time
import numpy as np
import os
import mujoco
import mujoco.viewer

MODEL_PATH = "button_scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home_key)
mujoco.mj_forward(model, data)

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTS = [f"act{i}" for i in range(1, 8)]
arm_qpos_idx = [model.joint(n).qposadr[0] for n in ARM_JOINTS]
arm_dof_idx = [model.joint(n).dofadr[0] for n in ARM_JOINTS]
arm_act_idx = [model.actuator(n).id for n in ARM_ACTS]
gripper_act_id = model.actuator("gripper").id

tcp_site = model.site("link_tcp").id
button_site = model.site("button_site").id
lamp_geom = model.geom("lamp_geom").id
touch_id = model.sensor("button_touch").id
touch_adr = model.sensor_adr[touch_id]

LAMP_OFF = np.array([0.3, 0.3, 0.3, 1.0])
LAMP_ON = np.array([1.0, 1.0, 0.2, 1.0])

home_qpos_arm = data.qpos[arm_qpos_idx].copy()
home_ctrl = data.ctrl.copy()

# Scratch data used ONLY for IK solves so we never disturb the live sim state.
ik_data = mujoco.MjData(model)

# Tracking flags for final summary (headless run).
button_pressed_ever = False
lamp_on_during_hold = False


def solve_ik(target_pos, seed_qpos, iters=150, tol=1e-3, step=0.5):
    """Damped least-squares IK for the 7 arm joints, holding TCP orientation fixed."""
    ik_data.qpos[:] = data.qpos
    ik_data.qpos[arm_qpos_idx] = seed_qpos
    mujoco.mj_forward(model, ik_data)
    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, ik_data.site_xmat[tcp_site])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    cur_quat = np.zeros(4)
    ori_err = np.zeros(3)
    for _ in range(iters):
        mujoco.mj_forward(model, ik_data)
        cur = ik_data.site_xpos[tcp_site]
        pos_err = target_pos - cur
        mujoco.mju_mat2Quat(cur_quat, ik_data.site_xmat[tcp_site])
        mujoco.mju_subQuat(ori_err, target_quat, cur_quat)
        err = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, tcp_site)
        jac_arm = np.vstack([jacp[:, arm_dof_idx], jacr[:, arm_dof_idx]])
        dq = np.linalg.pinv(jac_arm, rcond=1e-3) @ err
        ik_data.qpos[arm_qpos_idx] += step * dq
        for i, idx in enumerate(arm_qpos_idx):
            lo, hi = model.jnt_range[model.joint(ARM_JOINTS[i]).id]
            if hi > lo:
                ik_data.qpos[idx] = np.clip(ik_data.qpos[idx], lo, hi)
    return ik_data.qpos[arm_qpos_idx].copy()


def move_to(target_qpos_arm, duration_s, phase="", hold_gripper=0.0, track_lamp=False, viewer=None):
    """Smoothly ramp ctrl from current arm ctrl to target joint angles."""
    global lamp_on_during_hold
    start_ctrl = data.ctrl[arm_act_idx].copy()
    steps = max(1, int(duration_s / model.opt.timestep))
    if phase in ("retract_pullback", "retract", "return_home"):
        print_interval = 1
    else:
        print_interval = max(1, steps // 5)
    for s in range(steps):
        alpha = (s + 1) / steps
        data.ctrl[arm_act_idx] = start_ctrl + alpha * (target_qpos_arm - start_ctrl)
        data.ctrl[gripper_act_id] = hold_gripper
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        pressed = update_lamp()
        if track_lamp and pressed:
            lamp_on_during_hold = True
        if s % print_interval == 0 or s == steps - 1:
            touch_val = data.sensordata[touch_adr]
            lamp_state = "ON" if pressed else "OFF"
            print(f"[phase={phase}] step {s + 1}/{steps} touch={touch_val:.3f} lamp={lamp_state}")


def move_cartesian(start_pos, end_pos, seed_qpos, duration_s, phase="", hold_gripper=0.0, viewer=None):
    """Interpolate the TCP target in Cartesian space, solving IK fresh each step
    (warm-started from the previous solution) so the joint-space path never
    swings the TCP off the straight line between start_pos and end_pos."""
    steps = max(1, int(duration_s / model.opt.timestep))
    if phase in ("retract_pullback", "retract", "return_home"):
        print_interval = 1
    else:
        print_interval = max(1, steps // 5)
    q_cur = seed_qpos.copy()
    for s in range(steps):
        alpha = (s + 1) / steps
        target = start_pos + alpha * (end_pos - start_pos)
        q_cur = solve_ik(target, q_cur, iters=150)
        data.ctrl[arm_act_idx] = q_cur
        data.ctrl[gripper_act_id] = hold_gripper
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        pressed = update_lamp()
        if s % print_interval == 0 or s == steps - 1:
            touch_val = data.sensordata[touch_adr]
            lamp_state = "ON" if pressed else "OFF"
            print(f"[phase={phase}] step {s + 1}/{steps} touch={touch_val:.3f} lamp={lamp_state}")
    return q_cur


def update_lamp():
    global button_pressed_ever
    pressed = data.sensordata[touch_adr] > 0.5
    if pressed:
        button_pressed_ever = True
    model.geom_rgba[lamp_geom] = LAMP_ON if pressed else LAMP_OFF
    return pressed


def run(viewer=None):
    button_pos = data.site_xpos[button_site].copy()  # initial (unpressed) button center
    approach_pos = button_pos + np.array([-0.10, 0.0, 0.0])
    press_pos = button_pos + np.array([0.013, 0.0, 0.0])  # just past the button's 0.012m travel limit -- enough to fully depress it without overshooting into the panel and dragging the gripper

    update_lamp()

    # settle at home
    move_to(home_qpos_arm, 1.0, phase="home", viewer=viewer)

    # close gripper before approaching, staying at home position
    move_to(home_qpos_arm, 1.5, phase="close_gripper", hold_gripper=255.0, viewer=viewer)

    # approach above/in-front of the button
    q_approach = solve_ik(approach_pos, data.qpos[arm_qpos_idx])
    move_to(q_approach, 4.0, phase="approach", hold_gripper=255.0, viewer=viewer)

    # press the button
    q_press = solve_ik(press_pos, q_approach)
    move_to(q_press, 3.0, phase="press", hold_gripper=255.0, track_lamp=True, viewer=viewer)

    # hold press so the lamp is clearly seen glowing
    move_to(q_press, 5.0, phase="hold", hold_gripper=255.0, track_lamp=True, viewer=viewer)

    # pull straight back off the button along the press axis before retracting
    retreat_pos = press_pos + np.array([-0.05, 0.0, 0.05])
    q_retreat = move_cartesian(press_pos, retreat_pos, q_press, 1.5, phase="retract_pullback", hold_gripper=255.0, viewer=viewer)

    move_cartesian(retreat_pos, approach_pos, q_retreat, 3.0, phase="retract", hold_gripper=255.0, viewer=viewer)
    move_to(home_qpos_arm, 4.0, phase="return_home", hold_gripper=255.0, viewer=viewer)


if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        run(viewer=viewer)

        # let physics settle briefly at the end (headless, no rendering)
        settle_steps = int(1.0 / model.opt.timestep)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
            viewer.sync()
            update_lamp()

    final_qpos_arm = data.qpos[arm_qpos_idx]
    home_diff = np.linalg.norm(final_qpos_arm - home_qpos_arm)
    returned_home = home_diff < 0.05

    print("=" * 60)
    print("FINAL SUMMARY")
    print(f"  Button pressed:                    {button_pressed_ever}")
    print(f"  Lamp turned on during press/hold:  {lamp_on_during_hold}")
    print(f"  Returned to home (tol=0.05 rad, diff={home_diff:.4f}): {returned_home}")
    print("=" * 60)

    import sys; sys.stdout.flush(); os._exit(0)
