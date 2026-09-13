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
home_tcp_pos = data.site_xpos[tcp_site].copy()
button_site = model.site("button_site").id
lamp_geom = model.geom("lamp_geom").id
touch_id = model.sensor("button_touch").id
touch_adr = model.sensor_adr[touch_id]

LAMP_OFF = np.array([0.3, 0.3, 0.3, 1.0])
LAMP_ON = np.array([1.0, 1.0, 0.2, 1.0])

home_qpos_arm = data.qpos[arm_qpos_idx].copy()
home_ctrl = data.ctrl.copy()

# scratch data used ONLY for IK solves so we never disturb the live sim state.
ik_data = mujoco.MjData(model)

# tracking flags for final summary (headless run).
button_pressed_ever = False
lamp_on_during_hold = False


def solve_ik(target_pos, seed_qpos, iters=150, tol=1e-3, step=0.5):
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
    
    global lamp_on_during_hold #Smoothly ramp ctrl from current arm ctrl to target joint angles."
    start_ctrl = data.ctrl[arm_act_idx].copy()
    steps = max(1, int(duration_s / model.opt.timestep))
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
    global lamp_on_during_hold #Interpolate the TCP target in Cartesian space, solving IK fresh each step (warm-started from the previous solution) so the joint-space path never swings the TCP off the straight line between start_pos and end_pos."
    steps = max(1, int(duration_s / model.opt.timestep))
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
        if pressed:
            lamp_on_during_hold = True
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
    button_pos = data.site_xpos[button_site].copy()  # initial unpressed button center
    lifted_from_home = home_tcp_pos + np.array([0.0, 0.0, 0.15])
    approach_above = button_pos + np.array([0.0, 0.0, 0.15])
    top_of_button = button_pos + np.array([0.0, 0.0, 0.02])
    press_pos = button_pos - np.array([0.0, 0.0, 0.013])  # just past the button's 0.012m travel limit -- enough to fully depress it without overshooting into the panel and dragging the gripper
    retract_up = top_of_button + np.array([0.0, 0.0, 0.15])

    update_lamp()

    # settle at home
    move_to(home_qpos_arm, 1.0, phase="home", viewer=viewer)

    # close gripper before approaching, staying at home position
    move_to(home_qpos_arm, 1.5, phase="close_gripper", hold_gripper=255.0, viewer=viewer)

    # lift straight up off the home pose before moving over the button
    q_cur = data.qpos[arm_qpos_idx].copy()
    q_cur = move_cartesian(home_tcp_pos, lifted_from_home, q_cur, 1.5, phase="lift", hold_gripper=255.0, viewer=viewer)

    # move over to above the button, then descend to just above its surface
    q_cur = move_cartesian(lifted_from_home, approach_above, q_cur, 2.5, phase="move_to_button", hold_gripper=255.0, viewer=viewer)
    q_cur = move_cartesian(approach_above, top_of_button, q_cur, 1.5, phase="move_to_button", hold_gripper=255.0, viewer=viewer)

    # press the button straight down
    q_cur = move_cartesian(top_of_button, press_pos, q_cur, 1.5, phase="press", hold_gripper=255.0, viewer=viewer)

    # hold press so the lamp is clearly seen glowing
    q_cur = move_cartesian(press_pos, press_pos, q_cur, 5.0, phase="hold", hold_gripper=255.0, viewer=viewer)

    # retract straight up off the button and clear of the panel
    q_cur = move_cartesian(press_pos, top_of_button, q_cur, 1.0, phase="retract_up", hold_gripper=255.0, viewer=viewer)
    q_cur = move_cartesian(top_of_button, retract_up, q_cur, 2.0, phase="retract_up", hold_gripper=255.0, viewer=viewer)

    # return to home TCP position, then snap exactly to home joint angles, releasing the gripper
    # so the end-of-cycle gripper value matches the open (0.0) state the cycle starts from
    move_cartesian(retract_up, home_tcp_pos, q_cur, 3.0, phase="return_home", hold_gripper=0.0, viewer=viewer)
    move_to(home_qpos_arm, 4.0, phase="return_home", hold_gripper=0.0, viewer=viewer)


if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cycle = 0
        while viewer.is_running():
            cycle += 1
            run(viewer=viewer)

            # let physics settle briefly at the end of this cycle
            settle_steps = int(1.0 / model.opt.timestep)
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)
                viewer.sync()
                update_lamp()

            final_qpos_arm = data.qpos[arm_qpos_idx]
            home_diff = np.linalg.norm(final_qpos_arm - home_qpos_arm)
            returned_home = home_diff < 0.05

            print("=" * 60)
            print(f"CYCLE {cycle} SUMMARY")
            print(f"  Button pressed:                    {button_pressed_ever}")
            print(f"  Lamp turned on during press/hold:  {lamp_on_during_hold}")
            print(f"  Returned to home (tol=0.05 rad, diff={home_diff:.4f}): {returned_home}")
            print("=" * 60)

            if not viewer.is_running():
                break

            # cooldown before starting the next cycle
            time.sleep(2.0)

            # reset state for the next cycle
            mujoco.mj_resetDataKeyframe(model, data, home_key)
            mujoco.mj_forward(model, data)
            button_pressed_ever = False
            lamp_on_during_hold = False
            update_lamp()
