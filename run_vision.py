#!/usr/bin/env python3
"""Shared MuJoCo vision utilities and demo runner."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mujoco
import mujoco.viewer
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "universal_robots_ur10e", "scene.xml")
CAMERA_NAME = "vision_cam"
TARGET_BODY = "target_cube"
IMG_WIDTH = 640
IMG_HEIGHT = 480

# =========================
# Vision 总控台参数（中文）
# =========================
# 目标物体几何半边长（米）：仅用于深度补偿（表面深度 -> 物体中心深度）。
# 这是“尺寸参数”，不是“高度参数”。
TARGET_GEOM_HALF_EXTENT = 0.025

# 你只需要改这里：目标物体在世界坐标系中的“真实放置位置”（真值）。
# 相机本身并不知道这个真值；它仍然会通过图像与深度自行估计目标三维位置。
# 后续传给 task1/task2/task3 的位置，是视觉估计结果 detection.world_position。
TARGET_WORLD_X = 0.70
TARGET_WORLD_Y = 0.0
TARGET_WORLD_Z = 0.275
TARGET_WORLD_POSITION = np.array([TARGET_WORLD_X, TARGET_WORLD_Y, TARGET_WORLD_Z], dtype=float)

# 右下角小窗参数（MuJoCo 主窗口内叠加的相机画中画）。
OVERLAY_WIDTH_RATIO = 0.28
OVERLAY_MIN_WIDTH = 220
OVERLAY_MAX_WIDTH = 360
OVERLAY_MARGIN_PX = 16
OVERLAY_LABEL = CAMERA_NAME


@dataclass(frozen=True)
class VisionDetection:
    world_position: np.ndarray
    true_position: np.ndarray
    pixel: Tuple[int, int]
    surface_depth: float
    error_m: float
    rgb_image: np.ndarray
    annotated_rgb_image: np.ndarray


class ViewerWithCameraInset:
    """Proxy viewer that keeps a camera inset in the lower-right corner."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, viewer_handle, camera_name: str = CAMERA_NAME):
        self._model = model
        self._data = data
        self._viewer_handle = viewer_handle
        self._camera_name = camera_name
        self._renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

    def __getattr__(self, name):
        return getattr(self._viewer_handle, name)

    def close(self):
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            close_fn = getattr(renderer, "close", None)
            if callable(close_fn):
                close_fn()
            self._renderer = None

    def _compute_overlay_rect(self) -> Optional[mujoco.MjrRect]:
        viewport = self._viewer_handle.viewport
        if viewport is None:
            return None

        max_width = max(1, viewport.width - 2 * OVERLAY_MARGIN_PX)
        max_height = max(1, viewport.height - 2 * OVERLAY_MARGIN_PX)
        if max_width <= 1 or max_height <= 1:
            return None

        width = int(viewport.width * OVERLAY_WIDTH_RATIO)
        width = max(OVERLAY_MIN_WIDTH, width)
        width = min(OVERLAY_MAX_WIDTH, width, max_width)
        height = int(width * IMG_HEIGHT / IMG_WIDTH)

        if height > max_height:
            height = max_height
            width = int(height * IMG_WIDTH / IMG_HEIGHT)

        left = max(OVERLAY_MARGIN_PX, viewport.width - width - OVERLAY_MARGIN_PX)
        bottom = OVERLAY_MARGIN_PX
        return mujoco.MjrRect(left, bottom, max(1, width), max(1, height))

    def _render_overlay_image(self, width: int, height: int) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=self._camera_name)
        image = self._renderer.render().copy()
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)
        cv2.putText(
            image,
            OVERLAY_LABEL,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image

    def sync(self):
        rect = self._compute_overlay_rect()
        if rect is not None:
            image = self._render_overlay_image(rect.width, rect.height)
            self._viewer_handle.set_images((rect, image))
        self._viewer_handle.sync()


def get_camera_intrinsics(model: mujoco.MjModel, cam_id: int, width: int, height: int) -> Tuple[float, float, float]:
    fovy = model.cam_fovy[cam_id]
    focal = (height / 2.0) / np.tan(np.radians(fovy) / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return float(focal), float(cx), float(cy)


def _get_target_qpos_address(model: mujoco.MjModel, body_name: str = TARGET_BODY) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Cannot find body '{body_name}' in scene.")
    joint_adr = model.body_jntadr[body_id]
    if joint_adr < 0:
        raise ValueError(f"Body '{body_name}' does not have a joint.")
    return model.jnt_qposadr[joint_adr]


def set_target_body_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    world_position: np.ndarray = TARGET_WORLD_POSITION,
    body_name: str = TARGET_BODY,
) -> np.ndarray:
    target_xyz = np.asarray(world_position, dtype=float).reshape(3)
    qpos_adr = _get_target_qpos_address(model, body_name)
    data.qpos[qpos_adr:qpos_adr + 3] = target_xyz
    mujoco.mj_forward(model, data)
    return target_xyz.copy()


def _render_rgb_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = CAMERA_NAME,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    renderer: Optional[mujoco.Renderer] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    owned_renderer = renderer is None
    renderer = renderer or mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb_image = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        depth_image = renderer.render().copy()
        renderer.disable_depth_rendering()
        return rgb_image, depth_image
    finally:
        if owned_renderer:
            close_fn = getattr(renderer, "close", None)
            if callable(close_fn):
                close_fn()


def detect_target_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = CAMERA_NAME,
    target_body: str = TARGET_BODY,
    renderer: Optional[mujoco.Renderer] = None,
) -> VisionDetection:
    rgb_image, depth_image = _render_rgb_depth(model, data, camera_name=camera_name, renderer=renderer)

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Vision camera could not detect the yellow target cube.")

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise RuntimeError("Vision camera detected a degenerate target contour.")

    u = int(moments["m10"] / moments["m00"])
    v = int(moments["m01"] / moments["m00"])

    annotated = rgb_image.copy()
    cv2.circle(annotated, (u, v), 6, (0, 255, 0), -1)

    depth = float(depth_image[v, u])
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    focal, cx, cy = get_camera_intrinsics(model, cam_id, IMG_WIDTH, IMG_HEIGHT)

    center_depth = depth + TARGET_GEOM_HALF_EXTENT
    x_cam = (u - cx) * center_depth / focal
    y_cam = (cy - v) * center_depth / focal
    z_cam = -center_depth
    point_cam = np.array([x_cam, y_cam, z_cam], dtype=float)

    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    world_position = cam_pos + cam_mat @ point_cam

    true_position = data.body(target_body).xpos.copy()
    error_m = float(np.linalg.norm(world_position - true_position))

    return VisionDetection(
        world_position=world_position,
        true_position=true_position,
        pixel=(u, v),
        surface_depth=depth,
        error_m=error_m,
        rgb_image=rgb_image,
        annotated_rgb_image=annotated,
    )


def prepare_vision(model: mujoco.MjModel, data: mujoco.MjData) -> VisionDetection:
    """
    视觉准备入口：
    1) 先把目标物体放到 TARGET_WORLD_POSITION（真值，仅用于场景布置）；
    2) 再由相机图像 + 深度估计目标位置；
    3) 返回的 detection.world_position 才是给任务脚本使用的目标点。
    """
    set_target_body_position(model, data, TARGET_WORLD_POSITION)
    return detect_target_position(model, data)


def load_scene_with_target(xml_path: str = XML_PATH) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Cannot find MuJoCo scene: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    set_target_body_position(model, data, TARGET_WORLD_POSITION)
    return model, data


def print_detection_report(detection: VisionDetection):
    print("\n" + "=" * 48)
    print("Vision detection report")
    print("=" * 48)
    print(f"Configured target position : {TARGET_WORLD_POSITION}")
    print(f"Target geom half extent    : {TARGET_GEOM_HALF_EXTENT}")
    print(f"Image centroid (u, v)     : {detection.pixel}")
    print(f"Measured surface depth    : {detection.surface_depth:.4f} m")
    print(f"Detected world position   : {np.round(detection.world_position, 4)}")
    print(f"Physics true position     : {np.round(detection.true_position, 4)}")
    print(f"Spatial error             : {detection.error_m * 1000.0:.2f} mm")
    print("=" * 48)


def main():
    model, data = load_scene_with_target(XML_PATH)
    detection = detect_target_position(model, data)
    print_detection_report(detection)

    bgr_image = cv2.cvtColor(detection.annotated_rgb_image, cv2.COLOR_RGB2BGR)

    with mujoco.viewer.launch_passive(model, data) as viewer_handle:
        viewer = ViewerWithCameraInset(model, data, viewer_handle)
        try:
            while viewer_handle.is_running():
                viewer.sync()
                cv2.imshow("Robot Vision Camera", bgr_image)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                time.sleep(0.03)
        finally:
            viewer.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
