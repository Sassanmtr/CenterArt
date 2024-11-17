import numpy as np


def pose_4x4_to_flat(pose_4x4: np.ndarray) -> np.ndarray:
    rot = pose_4x4[:3, :3]
    trans = pose_4x4[:3, 3]
    pose_flat = np.concatenate([rot.flatten(), trans])
    return pose_flat


def pose_flat_to_4x4(pose_flat: np.ndarray) -> np.ndarray:
    rot = pose_flat[:9].reshape((3, 3))
    trans = pose_flat[9:]
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = rot
    pose_4x4[:3, 3] = trans
    return pose_4x4
