import argparse
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "scene.xml"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
ARM_ACTUATORS = ["act1", "act2", "act3", "act4", "act5", "act6", "act7"]

PEG_TIP_SITE = "peg_tip"
HOLE_ENTRY_SITE = "hole_entry"
HOLE_BOTTOM_SITE = "hole_bottom"

DAMPING = 1e-5
MAX_IK_ITERS = 400
POS_TOL = 1e-4


def solve_ik(model, data, site_name, target_pos, target_quat, q_init, joint_ids, dof_ids):
    data.qpos[dof_ids] = q_init #Damped least-squares IK for a site's position+orientation using only the arm joints.
    mujoco.mj_forward(model, data)
    site_id = model.site(site_name).id

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(MAX_IK_ITERS):
        mujoco.mj_forward(model, data)
        site_pos = data.site_xpos[site_id].copy()
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_mat.flatten())

        pos_err = target_pos - site_pos
        neg_quat = np.zeros(4)
        mujoco.mju_negQuat(neg_quat, site_quat)
        err_quat = np.zeros(4)
        mujoco.mju_mulQuat(err_quat, target_quat, neg_quat)
        quat_err = np.zeros(3)
        mujoco.mju_quat2Vel(quat_err, err_quat, 1.0)

        if np.linalg.norm(pos_err) < POS_TOL and np.linalg.norm(quat_err) < 1e-3:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        err = np.concatenate([pos_err, 0.5 * quat_err])

        JJt = J @ J.T + DAMPING * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 0.2:
            dq = dq * (0.2 / dq_norm)
        data.qpos[dof_ids] += dq
        for i, jid in enumerate(joint_ids):
            lo, hi = model.jnt_range[jid]
            if hi > lo:
                data.qpos[dof_ids[i]] = np.clip(data.qpos[dof_ids[i]], lo, hi)

    return data.qpos[dof_ids].copy()


def min_jerk(t):
    """Minimum-jerk scalar profile, t in [0,1] -> s in [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=False,
                         help="Run without the interactive viewer and print a pass/fail result.")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_ids = [model.joint(j).id for j in ARM_JOINTS]
    dof_ids = np.array([model.jnt_dofadr[j] for j in joint_ids])
    act_ids = [model.actuator(a).id for a in ARM_ACTUATORS]

    site_id_entry = model.site(HOLE_ENTRY_SITE).id
    site_id_bottom = model.site(HOLE_BOTTOM_SITE).id

    hole_entry_pos = data.site_xpos[site_id_entry].copy()
    hole_bottom_pos = data.site_xpos[site_id_bottom].copy()

    # Target orientation: peg axis vertical, pointing down into the hole.
    down_quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(down_quat, np.array([1.0, 0.0, 0.0]), np.pi)

    approach_pos = hole_entry_pos + np.array([0.0, 0.0, 0.08])

    q0 = data.qpos[dof_ids].copy()
    q_approach = solve_ik(model, data, PEG_TIP_SITE, approach_pos, down_quat, q0, joint_ids, dof_ids)
    data.qpos[dof_ids] = q_approach
    mujoco.mj_forward(model, data)
    peg_tip_id = model.site(PEG_TIP_SITE).id
    approach_residual = np.linalg.norm(data.site_xpos[peg_tip_id] - approach_pos)
    print(f"approach residual: {approach_residual:.4f}")

    q_entry = solve_ik(model, data, PEG_TIP_SITE, hole_entry_pos, down_quat, q_approach, joint_ids, dof_ids)
    data.qpos[dof_ids] = q_entry
    mujoco.mj_forward(model, data)
    peg_tip_id = model.site(PEG_TIP_SITE).id
    entry_residual = np.linalg.norm(data.site_xpos[peg_tip_id] - hole_entry_pos)
    print(f"entry residual: {entry_residual:.4f}")

    q_bottom = solve_ik(model, data, PEG_TIP_SITE, hole_bottom_pos, down_quat, q_entry, joint_ids, dof_ids)
    data.qpos[dof_ids] = q_bottom
    mujoco.mj_forward(model, data)
    peg_tip_id = model.site(PEG_TIP_SITE).id
    bottom_residual = np.linalg.norm(data.site_xpos[peg_tip_id] - hole_bottom_pos)
    print(f"bottom residual: {bottom_residual:.4f}")

    # Reset the simulation to the true initial state before animating.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    q_start = data.qpos[dof_ids].copy()

    waypoints = [q_start, q_approach, q_entry, q_bottom]
    phase_duration = 2.5  # seconds per phase

    if args.headless:
        phase = 0
        phase_time = 0.0
        while phase < len(waypoints) - 1:
            q_from = waypoints[phase]
            q_to = waypoints[phase + 1]
            s = min_jerk(phase_time / phase_duration)
            q_target = q_from + s * (q_to - q_from)
            for i, aid in enumerate(act_ids):
                data.ctrl[aid] = q_target[i]

            mujoco.mj_step(model, data)
            phase_time += model.opt.timestep

            if phase_time >= phase_duration:
                phase += 1
                phase_time = 0.0

        peg_tip_id = model.site(PEG_TIP_SITE).id
        final_dist = np.linalg.norm(data.site_xpos[peg_tip_id] - data.site_xpos[site_id_bottom])
        if final_dist < 0.01:
            print("INSERTION SUCCESS")
        else:
            print(f"INSERTION FAILED, distance={final_dist}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        phase = 0
        phase_time = 0.0
        while viewer.is_running() and phase < len(waypoints) - 1:
            q_from = waypoints[phase]
            q_to = waypoints[phase + 1]
            s = min_jerk(phase_time / phase_duration)
            q_target = q_from + s * (q_to - q_from)
            for i, aid in enumerate(act_ids):
                data.ctrl[aid] = q_target[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            phase_time += model.opt.timestep

            if phase_time >= phase_duration:
                phase += 1
                phase_time = 0.0

        # Hold the final inserted pose once all phases complete.
        while viewer.is_running():
            for i, aid in enumerate(act_ids):
                data.ctrl[aid] = q_bottom[i]
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
