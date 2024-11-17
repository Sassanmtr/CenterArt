# https://github.com/real-stanford/umpnet

import numpy as np

def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)

    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3, :3], xyz_pts.T)  # apply rotation
    xyz_pts = xyz_pts + np.tile(
        rigid_transform[:3, 3].reshape(3, 1), (1, xyz_pts.shape[1])
    )  # apply translation
    return xyz_pts.T


def get_ump_pointcloud(depth_img, color_img, segmentation_img, cam_intr, cam_pose=None):
    """Get 3D pointcloud from depth image.

    Args:
        depth_img: HxW float array of depth values in meters aligned with color_img
        color_img: HxWx3 uint8 array of color image
        segmentation_img: HxW int array of segmentation image
        cam_intr: 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix

    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
        color_pts: Nx3 uint8 array of color points
        color_pts: Nx1 int array of color points
    """

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x, pixel_y = np.meshgrid(
        np.linspace(0, img_w - 1, img_w), np.linspace(0, img_h - 1, img_h)
    )
    cam_pts_x = np.multiply(pixel_x - cam_intr[0, 2], depth_img / cam_intr[0, 0])
    cam_pts_y = np.multiply(pixel_y - cam_intr[1, 2], depth_img / cam_intr[1, 1])
    cam_pts_z = depth_img
    cam_pts = (
        np.array([cam_pts_x, cam_pts_y, cam_pts_z]).transpose(1, 2, 0).reshape(-1, 3)
    )
    # TODO: Commented for debugging
    if cam_pose is not None:
        cam_pts = transform_pointcloud(cam_pts, cam_pose)
    color_pts = None if color_img is None else color_img.reshape(-1, 3)
    segmentation_pts = (
        None if segmentation_img is None else segmentation_img.reshape(-1)
    )

    return cam_pts, color_pts, segmentation_pts


def get_position_action(affordance_map, epsilon, image, prev_actions):
    """Get position action based on affordance maps. (remove backgrund if rand() < 0.05)

    Returns:
        action: [w, h]s
        score: float
    """
    threshold = 0.1
    for prev_action in prev_actions:
        coord = image[prev_action[0], prev_action[1], :3]
        dist_map = np.sqrt(np.sum((image[:, :, :3] - coord) ** 2, axis=2))
        dist_mask = (dist_map > threshold).astype(float)
        affordance_map = affordance_map * dist_mask

    if np.random.rand() < epsilon or np.max(affordance_map) == 0:
        while True:
            idx = np.random.choice(affordance_map.size)
            action = np.array(np.unravel_index(idx, affordance_map.shape))
            z_value = image[action[0], action[1], 2]
            if z_value > 0.005 or np.random.rand() < 0.1:
                break
    else:
        idx = np.argmax(affordance_map)

    action = np.array(np.unravel_index(idx, affordance_map.shape))
    action = action.tolist()
    score = affordance_map[action[0], action[1]]

    return action, score
