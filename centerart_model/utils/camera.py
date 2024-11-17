import numpy as np
import open3d as o3d
from dataclasses import dataclass
from simnet.lib.camera import convert_homopixels_to_pixels, convert_points_to_homopoints

PYRENDER_INTRINSICS = np.array(
    [[627.4, 0.0, 325.5], [0.0, 627.4, 325.5], [0.0, 0.0, 1.0]]
)
REALSENSE_INTRINSICS = np.array(
    [[614.6375, 0.0, 313.9159], [0.0, 613.5278, 236.3182], [0.0, 0.0, 1.0]]
)
ZED2_INTRINSICS = np.array(
    [
        [1062.88232421875, 0.0, 957.660400390625],
        [0.0, 1062.88232421875, 569.8204345703125],
        [0.0, 0.0, 1.0],
    ]
)
TEST_INTRINSICS = np.array([[530.0, 0.0, 480.0], [0.0, 530.0, 256.0], [0.0, 0.0, 1.0]])

ZED2_INTRINSICS_HALF = np.copy(ZED2_INTRINSICS)
ZED2_INTRINSICS_HALF[0:-1, :] /= 2
ZED2_INTRINSICS_HALF[1, 2] -= 14  # Cropping
ZED2_RESOLUTION = np.array([1920, 1080], dtype=np.int32)
ZED2_RESOLUTION_HALF = ZED2_RESOLUTION // 2
ZED2_RESOLUTION_HALF[1] -= 28  # Cropping

PANDA_Q_START = np.array(
    [
        -1.4111348245688131,
        -1.2129180530427706,
        -0.13962625531258416,
        -2.358785666380864,
        0.46029318722089124,
        1.258500477120879,
        -0.5143043192269073,
    ]
)


def project(
    points_3d: np.ndarray, camera_intrinsics: np.ndarray = ZED2_INTRINSICS_HALF
):
    """
    Project a set of points in the camera frame onto the image plane

        points_3d: 3xN or 4xN

    return 2xN
    """
    if points_3d.shape[0] == 3:
        points_homo = convert_points_to_homopoints(points_3d)
    else:
        assert points_3d.shape[0] == 4
        points_homo = points_3d
    if camera_intrinsics.shape == (3, 3):
        camera_intrinsics = np.concatenate(
            (camera_intrinsics, np.zeros((3, 1))), axis=1
        )
    points_image = camera_intrinsics @ points_homo
    return convert_homopixels_to_pixels(points_image)


@dataclass
class CameraParamsIdeal:
    """
    This assumes ideal camera (i.e. cx, cy perfectly centered, no skew, fx = fy)
    """

    width: int
    height: int
    f_xy: float

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.f_xy, 0.0, self.cx], [0.0, self.f_xy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def fov_x_rad(self) -> float:
        return 2 * np.arctan2(self.width, 2 * self.f_xy)

    @property
    def fov_y_rad(self) -> float:
        return 2 * np.arctan2(self.height, 2 * self.f_xy)

    @property
    def fov_x_deg(self) -> float:
        return np.rad2deg(self.fov_x_rad)

    @property
    def fov_y_deg(self) -> float:
        return np.rad2deg(self.fov_y_rad)

    def to_open3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            width=self.width,
            height=self.height,
            fx=self.f_xy,
            fy=self.f_xy,
            cx=self.cx,
            cy=self.cy,
        )


def sample_cam_poses_shell(
    center: np.ndarray, coi_half_size: float, num_poses: int
) -> np.ndarray:
    """
    :param center: Center of the scene
    :param coi_half_size: Half size of the cube around the center of interest (cube of interest)
    :param num_poses: Number of poses to sample
    """
    poses_out = np.array(
        [sample_cam_pose_shell(center, coi_half_size) for _ in range(num_poses)]
    )
    return poses_out


def _normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

def look_at_z(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, -1])):
    """
    Camera looks in positive z-direction
    """
    eye = np.array(eye)
    target = np.array(target)
    zaxis = _normalize(target - eye)
    xaxis = _normalize(np.cross(up, zaxis))
    yaxis = _normalize(np.cross(zaxis, xaxis))
    m = np.eye(4)
    m[:3, 0] = xaxis
    m[:3, 1] = yaxis
    m[:3, 2] = zaxis
    m[:3, 3] = eye
    return m

def look_at_x(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])
):
    """
    Camera looks in positive x-direction
    """
    eye = np.array(eye)
    target = np.array(target)
    xaxis = _normalize(target - eye)
    yaxis = _normalize(np.cross(up, xaxis))
    zaxis = _normalize(np.cross(xaxis, yaxis))
    m = np.eye(4)
    m[:3, 0] = xaxis
    m[:3, 1] = yaxis
    m[:3, 2] = zaxis
    m[:3, 3] = eye
    return m


def sample_cam_pose_shell(center: np.ndarray, coi_half_size: float) -> np.ndarray:
    """
    :param center: Center of the scene
    :param coi_half_size: Half size of the cube around the center of interest (cube of interest)
    """
    point_of_interest = center + np.random.uniform(
        [-coi_half_size] * 3, [coi_half_size] * 3
    )
    # for umpnet data: elevation_min=45, elevation_max=65
    cam_position = sample_position_shell(
        center=point_of_interest,
        radius_min=2.1,
        radius_max=2.3,
        elevation_min=15,
        elevation_max=35,
        azimuth_min=-15,
        azimuth_max=15,
    )
    cam_pose = look_at_x(eye=cam_position, target=point_of_interest)
    return cam_pose


def sample_position_shell(
    center: np.ndarray,
    radius_min: float,
    radius_max: float,
    elevation_min: float = 0,
    elevation_max: float = 90,
    azimuth_min: float = -60,
    azimuth_max: float = 60,
) -> np.ndarray:
    """
    Samples a point from the volume between two spheres (radius_min, radius_max). Optionally the
    spheres can be constraint by setting elevation and azimuth angles. E.g. if you only want to
    sample in the upper hemisphere set elevation_min = 0. Instead of sampling the angles and radius
    uniformly, sample the shell volume uniformly. As a result, there will be more samples at larger
    radii.

    :param center: Center shared by both spheres.
    :param radius_min: Radius of the smaller sphere.
    :param radius_max: Radius of the bigger sphere.
    :param elevation_min: Minimum angle of elevation in degrees. Range: [-90, 90].
    :param elevation_max: Maximum angle of elevation in degrees. Range: [-90, 90].
    :param azimuth_min: Minimum angle of azimuth in degrees. Range: [-180, 180].
    :param azimuth_max: Maximum angle of azimuth in degrees. Range: [-180, 180].
    :return: A sampled point.
    """
    assert -180 <= azimuth_min <= 180, "azimuth_min must be in range [-180, 180]"
    assert -180 <= azimuth_max <= 180, "azimuth_max must be in range [-180, 180]"
    assert -90 <= elevation_min <= 90, "elevation_min must be in range [-90, 90]"
    assert -90 <= elevation_min <= 90, "elevation_max must be in range [-90, 90]"
    assert azimuth_min < azimuth_max, "azimuth_min must be smaller than azimuth_max"
    assert (
        elevation_min < elevation_max
    ), "elevation_min must be smaller than elevation_max"

    radius = radius_min + (radius_max - radius_min) * np.cbrt(np.random.rand())

    # rejection sampling
    constr_fulfilled = False
    while not constr_fulfilled:
        direction_vector = np.random.randn(3)
        direction_vector /= np.linalg.norm(direction_vector)

        # https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
        xy = (
            direction_vector[0] * direction_vector[0]
            + direction_vector[1] * direction_vector[1]
        )
        elevation = np.arctan2(direction_vector[2], np.sqrt(xy))
        azimuth = np.arctan2(direction_vector[1], direction_vector[0])

        elev_constraint = (
            np.deg2rad(elevation_min) < elevation < np.deg2rad(elevation_max)
        )
        azim_constraint = np.deg2rad(azimuth_min) < azimuth < np.deg2rad(azimuth_max)
        constr_fulfilled = elev_constraint and azim_constraint

    # Get the coordinates of a sampled point inside the shell
    position = direction_vector * radius + center

    return position


@dataclass
class CameraConventions:
    """
    Different libraries use different camera coordinate conventions:
    - OpenGL/Blender: +X is right, +Y is up, and +Z is pointing back
    - OpenCV/COLMAP: +X is right, +Y is down, and +Z is pointing forward
    - Robotics/SAPIEN: +X is forward, +Y is left, +Z is up
    """

    opengl_T_opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ).T
    opengl_T_robotics = np.array(
        [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    ).T
    opencv_T_robotics = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    ).T
    z_T_z = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_T_y = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    x_T_x = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    opencv_T_opengl = opengl_T_opencv.T
    robotics_T_opengl = opengl_T_robotics.T
    robotics_T_opencv = opencv_T_robotics.T
    opengl_R_opencv = opengl_T_opencv[:3, :3]
    opengl_R_robotics = opengl_T_robotics[:3, :3]
    opencv_R_robotics = opencv_T_robotics[:3, :3]
    opencv_R_opengl = opencv_T_opengl[:3, :3]
    robotics_R_opencv = robotics_T_opencv[:3, :3]
    robotics_R_opengl = robotics_T_opengl[:3, :3]
