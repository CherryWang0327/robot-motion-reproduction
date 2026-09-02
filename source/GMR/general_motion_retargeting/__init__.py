"""GMR public API.

Heavy MuJoCo/IK dependencies are imported lazily so data conversion utilities can
be used in lightweight environments.
"""

__all__ = [
    "GeneralMotionRetargeting", "RobotMotionViewer", "draw_frame",
    "load_robot_motion", "KinematicsModel", "human_head_to_robot_neck",
    "XRobotStreamer", "XRobotRecorder",
    "IK_CONFIG_ROOT", "ASSET_ROOT", "ROBOT_XML_DICT", "IK_CONFIG_DICT",
    "ROBOT_BASE_DICT", "VIEWER_CAM_DISTANCE_DICT",
]


def __getattr__(name):
    parameter_names = {
        "IK_CONFIG_ROOT", "ASSET_ROOT", "ROBOT_XML_DICT", "IK_CONFIG_DICT",
        "ROBOT_BASE_DICT", "VIEWER_CAM_DISTANCE_DICT",
    }
    if name in parameter_names:
        from . import params
        return getattr(params, name)
    if name == "GeneralMotionRetargeting":
        from .motion_retarget import GeneralMotionRetargeting
        return GeneralMotionRetargeting
    if name in ("RobotMotionViewer", "draw_frame"):
        from .robot_motion_viewer import RobotMotionViewer, draw_frame
        return {"RobotMotionViewer": RobotMotionViewer, "draw_frame": draw_frame}[name]
    if name == "load_robot_motion":
        from .data_loader import load_robot_motion
        return load_robot_motion
    if name == "KinematicsModel":
        from .kinematics_model import KinematicsModel
        return KinematicsModel
    if name == "human_head_to_robot_neck":
        from .neck_retarget import human_head_to_robot_neck
        return human_head_to_robot_neck
    if name in ("XRobotStreamer", "XRobotRecorder"):
        try:
            from .xrobot_utils import XRobotRecorder, XRobotStreamer
        except ImportError:
            return None
        return {"XRobotStreamer": XRobotStreamer, "XRobotRecorder": XRobotRecorder}[name]
    raise AttributeError(name)
