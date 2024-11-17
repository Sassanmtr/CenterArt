import numpy as np
import trimesh

def find_closest(points, successful_grasps):
    """
    Find the closest 3d point for each grasp pose, and return the indeces.
    Input arrays must be np arrays of shape (q, 3) and (c, 3) respectively.
    """
    diff = successful_grasps[None, ...] - points[:, None, ...]
    dist = np.linalg.norm(diff, axis=-1)
    min_idxs = np.argmin(dist, axis=-1)
    min_vals = dist[np.arange(len(min_idxs)), min_idxs]
    return min_idxs, min_vals


def get_gf_v_vec():
    '''
    This function calculates and returns the vertices of a gripper in 3-dimensional space. It assumes that the 
    gripper fingers are symmetric around the y-axis.

    The returned numpy array represents five points of the gripper fingers: the midpoint, left and right points
    at the middle of the gripper width, and left and right points at the top of the gripper width.
    '''
    gripper_width = 0.08
    left_x = -0.5 * gripper_width
    right_x = 0.5 * gripper_width
    mid_z = 0.066
    top_z = 0.112
    a = [0, 0, 0]
    b = [left_x, 0, mid_z]
    c = [right_x, 0, mid_z]
    d = [left_x, 0, top_z]
    e = [right_x, 0, top_z]
    gf_v = np.array([a, b, c, d, e]) - np.array([0, 0, 0.075])
    return gf_v


def v_vec_from_grasp(grasp_pose):
    '''
    Transforms gripper finger vertices from grasp frame to world frame
    '''
    # v in grasp frame
    gf_v = get_gf_v_vec()
    # v in world frame
    wf_v = np.array([grasp_pose @ np.append(x, 1) for x in gf_v])[:, :3]
    return wf_v


def generate_gt_v(xyz_points, successful_grasps):
    '''
    Finds the closest successful grasp for each point in `xyz_points` and transforms the vertices of the
    gripper fingers in the world frame for each of these closest successful grasps.
    '''
    min_idxs, _ = find_closest(xyz_points, successful_grasps[:, :3, 3])
    closest_grasps = successful_grasps[min_idxs]
    v_vecs = [v_vec_from_grasp(grasp_pose) for grasp_pose in closest_grasps]
    return v_vecs


def argmax_n(ary, n):
    """Returns the indices of the n largest elements from a numpy array."""
    n = min(n, len(ary))
    indices = np.argpartition(ary, -n)[-n:]
    values = ary[indices]
    return indices, values


def get_best_grasps(s_confidence, grasp_poses, n=10):
    n_min_idxs, _ = argmax_n(s_confidence, n)
    best_grasps = grasp_poses[n_min_idxs]
    return best_grasps

def create_gripper_marker(
    color=[0, 0, 255], gripper_width=0.08, tube_radius=0.002, sections=6
):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    left_y = -0.5 * (gripper_width + tube_radius)
    right_y = 0.5 * (gripper_width + tube_radius)
    mid_z = 0.066
    top_z = 0.112
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [0.0, right_y, mid_z],
            [0.0, right_y, top_z],
        ],
    )
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [0.0, left_y, mid_z],
            [0.0, left_y, top_z],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, 0], [0, 0, mid_z]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[0.0, left_y, mid_z], [0.0, right_y, mid_z]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def create_markers(transform, color, tube_radius=0.005, axis_frame: bool = True):
    original_transform = transform.copy()
    original_transform[:3, 3] -= 0.075 * transform[:3, 2]
    position_marker = trimesh.creation.axis(transform=transform, origin_size=0.005)  # type: ignore
    stick_marker = create_gripper_marker(color, tube_radius=tube_radius)  # type: ignore
    stick_marker.apply_transform(original_transform)  # type: ignore
    return (
        np.array([stick_marker] + ([position_marker] if axis_frame else []))
        .flatten()
        .tolist()
    )

def create_markers_multiple(
    transforms, color, axis_frame: bool = True, highlight_first: bool = False
):
    res = (
        np.array([create_markers(t, color, axis_frame=axis_frame) for t in transforms])
        .flatten()
        .tolist()
    )
    # if highlight_first and len(transforms) > 0:
    if (
        highlight_first and len(transforms) > 1
    ):  # changed to 1 to highlight only the first grasp
        first_marker = create_markers(
            transforms[0], color, tube_radius=0.01, axis_frame=axis_frame
        )
        res[0] = first_marker[0]
    return res


def create_markers_multiple_fat(
    hand_poses: np.ndarray, color: list, axis_frame: bool = True
):
    res = (
        np.array(
            [
                create_markers(t, color, tube_radius=0.01, axis_frame=axis_frame)
                for t in hand_poses
            ]
        )
        .flatten()
        .tolist()
    )
    return res
