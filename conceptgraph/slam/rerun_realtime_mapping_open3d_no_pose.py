# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/dense_slam_gui.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.
# "pip install numpy==1.24.3" to avoid "Segmentation Fault" error

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import hydra
from conceptgraph.utils.config import ConfigParser
from conceptgraph.dataset import conceptgraphs_datautils

import os
import numpy as np
import threading
import time
from omegaconf import DictConfig
import argparse
from conceptgraph.utils.common import load_rgbd_file_names, save_poses, load_intrinsic, extract_trianglemesh, get_default_dataset, extract_rgbd_frames
from conceptgraph.utils import general_utils

###################
# Standard library imports
import shutil
import pickle
import gzip
import yaml
# Third-party imports
import torch
import PIL
from tqdm import trange
# from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import Counter
# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, orr_log_camera, orr_log_depth_image,
    orr_log_edges, orr_log_objs_pcd_and_bbox, orr_log_rgb_image,
    orr_log_vlm_image)

from conceptgraph.utils.logging_metrics import MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, get_det_out_path, get_exp_out_path,
    get_vlm_annotated_image_path, make_vlm_edges_and_captions, measure_time,
    save_detection_results, save_edge_json, save_hydra_config, save_obj_json,
    save_objects_for_frame, save_pointcloud)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
                                    vis_result_fast_on_depth,vis_result_fast,
                                    save_video_detections)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs, filter_objects, get_bounding_box, init_process_pcd,
    make_detection_list_from_pcd_and_gobs, denoise_objects, merge_objects,
    detections_to_obj_pcd_and_bbox, process_cfg,
    process_edges, processing_needed, resize_gobs)
from conceptgraph.slam.mapping import (compute_spatial_similarities,
                                       compute_visual_similarities,
                                       aggregate_similarities,
                                       match_detections_to_objects,
                                       merge_obj_matches)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections
import cv2
from typing import Tuple, List, Dict, Any, Union, Optional
import argparse
from scipy.spatial.transform import Rotation as R
import os
import numpy as np

# Disable torch gradient computation
torch.set_grad_enabled(False)
RUN_OPEN_API = False
RUN_START = False
RUN_MIDDLE = False
RUN_AFTER = False


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable


class ReconstructionWindow:

    def __init__(self, config: ConfigParser, font_id: int, cfg: DictConfig, args: argparse.Namespace):
        ####################3
        self.common_init(cfg, args)
        self.dataset = get_dataset(
            # dataset/dataconfigs/replica/replica.yaml
            dataconfig=cfg.dataset_config,
            # Replica
            basedir=cfg.dataset_root,
            # room0
            sequence=cfg.scene_id,
            start=cfg.start,  # 0
            end=cfg.end,  # -1
            stride=1, #cfg.stride,  # 50
            desired_height=int(cfg.image_height * self.args.resize_ratio),  # None # 680
            desired_width=int(cfg.image_width * self.args.resize_ratio),  # None # 1200
            device="cpu",
            dtype=torch.float,
        )
        # self.dataset_init(cfg)


        self.config = config
        self.window = gui.Application.instance.create_window(
            'Open3D - Reconstruction', 1280, 800)
        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)
        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Items in fixed props
        self.fixed_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))

        ### Depth scale slider
        scale_label = gui.Label('Depth scale')
        self.scale_slider = gui.Slider(gui.Slider.INT)
        self.scale_slider.set_limits(1000, 7000)
        self.scale_slider.int_value = int(config.depth_scale)
        self.fixed_prop_grid.add_child(scale_label)
        self.fixed_prop_grid.add_child(self.scale_slider)
        voxel_size_label = gui.Label('Voxel size')
        self.voxel_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_size_slider.set_limits(0.003, 0.01)
        self.voxel_size_slider.double_value = config.voxel_size
        self.fixed_prop_grid.add_child(voxel_size_label)
        self.fixed_prop_grid.add_child(self.voxel_size_slider)

        trunc_multiplier_label = gui.Label('Trunc multiplier')
        self.trunc_multiplier_slider = gui.Slider(gui.Slider.DOUBLE)
        self.trunc_multiplier_slider.set_limits(1.0, 20.0)
        self.trunc_multiplier_slider.double_value = config.trunc_voxel_multiplier
        self.fixed_prop_grid.add_child(trunc_multiplier_label)
        self.fixed_prop_grid.add_child(self.trunc_multiplier_slider)

        est_block_count_label = gui.Label('Est. blocks')
        self.est_block_count_slider = gui.Slider(gui.Slider.INT)
        self.est_block_count_slider.set_limits(4000, 100000)
        self.est_block_count_slider.int_value = config.block_count
        self.fixed_prop_grid.add_child(est_block_count_label)
        self.fixed_prop_grid.add_child(self.est_block_count_slider)
        est_point_count_label = gui.Label('Est. points')
        self.est_point_count_slider = gui.Slider(gui.Slider.INT)
        self.est_point_count_slider.set_limits(500000, 48000000)
        self.est_point_count_slider.int_value = config.est_point_count
        self.fixed_prop_grid.add_child(est_point_count_label)
        self.fixed_prop_grid.add_child(self.est_point_count_slider)

        ## Items in adjustable props
        self.adjustable_prop_grid = gui.VGrid(2, spacing,
                                              gui.Margins(em, 0, em, 0))

        ### Reconstruction interval
        interval_label = gui.Label('Recon. interval')
        self.interval_slider = gui.Slider(gui.Slider.INT)
        self.interval_slider.set_limits(1, 500)
        self.interval_slider.int_value = 50
        self.adjustable_prop_grid.add_child(interval_label)
        self.adjustable_prop_grid.add_child(self.interval_slider)

        ### Depth max slider
        max_label = gui.Label('Depth max')
        self.max_slider = gui.Slider(gui.Slider.DOUBLE)
        self.max_slider.set_limits(3.0, 6.0)
        self.max_slider.double_value = config.depth_max
        self.adjustable_prop_grid.add_child(max_label)
        self.adjustable_prop_grid.add_child(self.max_slider)

        ### Depth diff slider
        diff_label = gui.Label('Depth diff')
        self.diff_slider = gui.Slider(gui.Slider.DOUBLE)
        self.diff_slider.set_limits(0.07, 0.5)
        self.diff_slider.double_value = config.odometry_distance_thr
        self.adjustable_prop_grid.add_child(diff_label)
        self.adjustable_prop_grid.add_child(self.diff_slider)
        ### Update surface?
        update_label = gui.Label('Update surface?')
        self.update_box = gui.Checkbox('')
        self.update_box.checked = True
        self.adjustable_prop_grid.add_child(update_label)
        self.adjustable_prop_grid.add_child(self.update_box)

        ### Ray cast color?
        raycast_label = gui.Label('Raycast color?')
        self.raycast_box = gui.Checkbox('')
        self.raycast_box.checked = True
        self.adjustable_prop_grid.add_child(raycast_label)
        self.adjustable_prop_grid.add_child(self.raycast_box)

        set_enabled(self.fixed_prop_grid, True)
        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)
        ### Rendered image tab
        tab2 = gui.Vert(0, tab_margins)
        self.raycast_color_image = gui.ImageWidget()
        self.raycast_depth_image = gui.ImageWidget()
        tab2.add_child(self.raycast_color_image)
        tab2.add_fixed(vspacing)
        tab2.add_child(self.raycast_depth_image)
        tabs.add_tab('Raycast images', tab2)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        self.panel.add_child(gui.Label('Starting settings'))
        self.panel.add_child(self.fixed_prop_grid)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(gui.Label('Reconstruction settings'))
        self.panel.add_child(self.adjustable_prop_grid)
        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)

        # Scene widget
        self.widget3d = gui.SceneWidget()
        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)
        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)
        self.is_done = False

        self.is_started = False
        self.is_running = False
        self.is_surface_updated = False

        self.idx = 0
        self.poses = []

        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def common_init(self, cfg: DictConfig, args: argparse.Namespace):

        self.cfg = cfg
        self.args = args
        self.extrinsic = None #self._set_extrinsic()
        # tracker : **탐지된 객체**, **병합된 객체** 및 **운영 수**와 같은 여러 상태 정보를 관리
        self.tracker = MappingTracker()
        # 만약 Rerun이 설치되어 있지 않거나, 사용하지 않는 경우, 이 변수는 None입니다.
        self.orr = OptionalReRun()
        self.orr.set_use_rerun(self.cfg.use_rerun)  # True
        self.orr.init("realtime_mapping")
        self.orr.spawn()

        self.cfg = process_cfg(self.cfg)
        self.objects = MapObjectList(device=self.cfg.device)
        self.map_edges = MapEdgeMapping(self.objects)


        exp_path = "./Datasets/Replica/room0/exps"
        if os.path.exists(exp_path):
            user_input = input(
                f"The folder {exp_path} already exists. Do you want to delete it? (y/n): "
            ).strip().lower()

            if user_input == 'y':
                # 폴더 삭제
                shutil.rmtree(exp_path)
                print(f"The folder {exp_path} has been deleted.")
            else:
                print("The folder has not been deleted.")
        # output folder for this mapping experiment
        # dataset_root: Datasets
        # scene_id: Replica/room0
        # exp_suffix: r_mapping_stride10
        # self.exp_out_path: Datasets/Replica/room0/exps/r_mapping_stride10
        self.exp_out_path = get_exp_out_path(self.cfg.dataset_root,
                                             self.cfg.scene_id,
                                             self.cfg.exp_suffix)

        # output folder of the detections experiment to use
        # det_exp_path: Datasets/Replica/room0/exps/s_detections_stride10
        self.det_exp_path = get_exp_out_path(self.cfg.dataset_root,
                                             self.cfg.scene_id,
                                             self.cfg.detections_exp_suffix,
                                             make_dir=False)

        # we need to make sure to use the same classes as the ones used in the detections
        detections_exp_cfg = cfg_to_dict(self.cfg)
        # obj_classes 에서 person 제외했음
        self.obj_classes = ObjectClasses(
            classes_file_path=detections_exp_cfg['classes_file'],
            bg_classes=detections_exp_cfg['bg_classes'],
            skip_bg=detections_exp_cfg['skip_bg'])

        # if we need to do detections
        # det_exp_path:
        # concept-graphs/Datasets/Replica/room0/exps/s_detections_stride10
        # check_run_detections: s_detections_stride10 폴더가 있으면 실시하지 않음
        self.run_detections = check_run_detections(self.cfg.force_detection,
                                                   self.det_exp_path)
        self.run_detections = True
        # det_exp_pkl_path: exps/s_detections_stride10/detections
        self.det_exp_pkl_path = get_det_out_path(self.det_exp_path)
        # det_exp_vis_path: exps/s_detections_stride10/vis
        self.det_exp_vis_path = get_vis_out_path(self.det_exp_path)

        self.prev_adjusted_pose = None

        if self.run_detections:
            print("\n".join(["Running detections..."] * 10))
            self.det_exp_path.mkdir(parents=True, exist_ok=True)

            ## Initialize the detection models
            self.detection_model = measure_time(YOLO)('yolov8x-worldv2.pt')# ('yolov8l-world.pt')#
            self.sam_predictor = SAM(
                'sam_b.pt')  # SAM('mobile_sam.pt') # UltraLytics SAM
            (self.clip_model, _,
             self.clip_preprocess) = open_clip.create_model_and_transforms(
                 "ViT-H-14", "laion2b_s32b_b79k")
            self.clip_model = self.clip_model.to(self.cfg.device)
            self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

            # Set the classes for the detection model
            self.detection_model.set_classes(self.obj_classes.get_classes_arr())
            if RUN_OPEN_API:
                self.openai_client = get_openai_client()
            else:
                self.openai_client = None

        else:
            print("\n".join(["NOT Running detections..."] * 10))

        save_hydra_config(self.cfg, self.exp_out_path)
        save_hydra_config(detections_exp_cfg,
                          self.exp_out_path,
                          is_detection_config=True)

        if self.cfg.save_objects_all_frames:
            # exp_out_path: Datasets/Replica/room0/exps/r_mapping_stride10
            # obj_all_frames_out_path: room0/exps/r_mapping_stride10/saved_obj_all_frames/det_s_detections_stride10
            self.obj_all_frames_out_path = (
                self.exp_out_path / "saved_obj_all_frames" /
                f"det_{self.cfg.detections_exp_suffix}")
            os.makedirs(self.obj_all_frames_out_path, exist_ok=True)

        self.counter = 0
        self.frame_idx = -1


    # def _set_extrinsic(self) -> np.ndarray:
    #     axis_transpose_matrix = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
    #                                       [0., -1., 0., 0.], [0., 0., 0., 1.]])
    #     camera_translation = np.array([0.37, 0.035, 0.862])
    #     camera_radian_rotation = np.deg2rad(np.array([0., 0., 0.]))
    #     rotation = R.from_euler('xyz', camera_radian_rotation)
    #     rotation_matrix_3 = rotation.as_matrix()
    #     rotation_matrix = np.eye(4)
    #     rotation_matrix[:3, :3] = rotation_matrix_3
    #     translation_matrix = tf_transformations.translation_matrix(
    #         camera_translation)
    #     (camera_pose_wrt_agent
    #     ) = translation_matrix @ rotation_matrix @ axis_transpose_matrix
    #     return camera_pose_wrt_agent

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                        rect.y, fps_panel_width,
                                        fps_panel_height)

    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)
        self.is_running = not self.is_running

    # On start: point cloud buffer and model initialization.
    def _on_start(self):
        max_points = self.est_point_count_slider.int_value

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
        pcd_placeholder.point.colors = o3c.Tensor(
            np.zeros((max_points, 3), dtype=np.float32))
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)

        self.model = o3d.t.pipelines.slam.Model(
            self.voxel_size_slider.double_value, 16,
            self.est_block_count_slider.int_value, o3c.Tensor(np.eye(4)),
            o3c.Device(self.config.device))
        self.is_started = True

        set_enabled(self.fixed_prop_grid, False)
        set_enabled(self.adjustable_prop_grid, True)

    def _on_close(self):
        self.is_done = True

        if self.is_started:
            print('Saving model to {}...'.format(self.config.path_npz))
            self.model.voxel_grid.save(self.config.path_npz)
            print('Finished.')

            mesh_fname = '.'.join(self.config.path_npz.split('.')[:-1]) + '.ply'
            print('Extracting and saving mesh to {}...'.format(mesh_fname))
            mesh = extract_trianglemesh(self.model.voxel_grid, self.config,
                                        mesh_fname)
            print('Finished.')

            log_fname = '.'.join(self.config.path_npz.split('.')[:-1]) + '.log'
            print('Saving trajectory to {}...'.format(log_fname))
            save_poses(log_fname, self.poses)
            print('Finished.')
            self.wrap_up()
        return True

    def init_render(self, depth_ref, color_ref):
        """
        `init_render` 메서드는 초기 렌더링 설정을 담당하며,
            주로 첫 번째 깊이 이미지와 컬러 이미지를 사용해 3D 장면을 초기화
        이 메서드는 `Open3D`의 GUI를 활용해 시각화하는데, 다음과 같은 주요 단계로 나눌 수 있습니다:

        1. **깊이 및 컬러 이미지 업데이트:**
           - 첫 번째로 주어진 깊이 및 컬러 이미지를 사용하여 GUI의 이미지 위젯을 업데이트
           - 이 작업은 사용자가 처음 GUI를 볼 때 깊이와 컬러 데이터를 시각적으로 확인할 수 있도록

        2. **깊이 데이터 시각화:**
           - 깊이 이미지는 `colorize_depth`라는 함수를 사용하여 색상으로 변환
           - 이를 통해 깊이 데이터의 시각적인 표현이 가능

        3. **카메라 설정:**
           - 가상 카메라의 위치와 뷰포인트를 설정
           - 이를 위해 축에 정렬된 바운딩 박스(`AxisAlignedBoundingBox`)를 사용하여
             - 3D 공간의 크기와 위치를 지정하고,
        - 카메라의 뷰를 중심점으로 이동시켜 사용자가 3D 장면을 적절하게 관찰할 수 있도록 합니다.
           - 카메라의 시야각을 설정하고 장면의 초기 뷰를 정합니다.

        4. **초기 렌더링 준비:**
           - 위의 단계에서 설정된 이미지를 GUI에 적용하고, 사용자 인터페이스를 업데이트
           - 이를 통해 사용자는 첫 번째 프레임을 확인할 수 있고,
             - 이후 프레임이 추가적으로 처리될 때의 기준점이 만들어집니다.

        """
        self.input_depth_image.update_image(
            depth_ref.colorize_depth(float(self.scale_slider.int_value),
                                     self.config.depth_min,
                                     self.max_slider.double_value).to_legacy())
        self.input_color_image.update_image(color_ref.to_legacy())

        self.raycast_depth_image.update_image(
            depth_ref.colorize_depth(float(self.scale_slider.int_value),
                                     self.config.depth_min,
                                     self.max_slider.double_value).to_legacy())
        self.raycast_color_image.update_image(color_ref.to_legacy())
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

    def update_render(self, input_depth, input_color, raycast_depth,
                      raycast_color, pcd, frustum):
        self.input_depth_image.update_image(
            input_depth.colorize_depth(
                float(self.scale_slider.int_value), self.config.depth_min,
                self.max_slider.double_value).to_legacy())
        self.input_color_image.update_image(input_color.to_legacy())

        self.raycast_depth_image.update_image(
            raycast_depth.colorize_depth(
                float(self.scale_slider.int_value), self.config.depth_min,
                self.max_slider.double_value).to_legacy())
        self.raycast_color_image.update_image(
            (raycast_color).to(o3c.uint8, False, 255.0).to_legacy())

        if self.is_scene_updated:
            if pcd is not None and pcd.point.positions.shape[0] > 0:
                self.widget3d.scene.scene.update_geometry(
                    'points', pcd, rendering.Scene.UPDATE_POINTS_FLAG |
                    rendering.Scene.UPDATE_COLORS_FLAG)

        self.widget3d.scene.remove_geometry("frustum")
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum", frustum, mat)

    # Major loop
    def update_main(self):
        depth_file_names, color_file_names = load_rgbd_file_names(self.config)

        # intrinsic: intrinsic_matrix Tensor
        intrinsic = load_intrinsic(self.config)
        intrinsic[0,0] *= self.args.resize_ratio
        intrinsic[1,1] *= self.args.resize_ratio
        intrinsic[0,2] *= self.args.resize_ratio
        intrinsic[1,2] *= self.args.resize_ratio

        n_files = len(color_file_names)
        # traj_path = os.path.join(self.config.path_dataset, "traj.txt")
        # if os.path.exists(traj_path):
        #     loaded_poses = general_utils.load_poses(traj_path, n_files)
        #     print("Loaded poses from {}".format(traj_path))
        # else:
        T_frame_to_model = o3c.Tensor(np.identity(4))
        device = o3d.core.Device(self.config.device)
        if self.args.resize_ratio < 1.:
            depth_ref = o3d.t.io.read_image(depth_file_names[0]).resize(self.args.resize_ratio)
            color_ref = o3d.t.io.read_image(color_file_names[0]).resize(self.args.resize_ratio)
        elif self.args.resize_ratio == 1.:
            depth_ref = o3d.t.io.read_image(depth_file_names[0])
            color_ref = o3d.t.io.read_image(color_file_names[0])
        else:
            raise ValueError("Invalid resize ratio")
        input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                 depth_ref.columns, intrinsic,
                                                 device)
        raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                   depth_ref.columns, intrinsic,
                                                   device)

        input_frame.set_data_from_image('depth', depth_ref)
        input_frame.set_data_from_image('color', color_ref)

        raycast_frame.set_data_from_image('depth', depth_ref)
        raycast_frame.set_data_from_image('color', color_ref)

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render(depth_ref, color_ref))
        fps_interval_len = 30
        self.idx = 0
        pcd = None

        ##########################################
        np.random.seed(42)
        avg_fixed_minus_gt_deg = np.zeros(6)
        avg_noise_deg = np.zeros(6)
        ##########################################

        start = time.time()
        while not self.is_done:
            if not self.is_started or not self.is_running:
                time.sleep(0.05)
                continue
            if self.args.resize_ratio < 1.:
                depth = o3d.t.io.read_image(depth_file_names[self.idx]).resize(self.args.resize_ratio).to(device)
                color = o3d.t.io.read_image(color_file_names[self.idx]).resize(self.args.resize_ratio).to(device)
            elif self.args.resize_ratio == 1.:
                depth = o3d.t.io.read_image(depth_file_names[self.idx]).to(device)
                color = o3d.t.io.read_image(color_file_names[self.idx]).to(device)
            else:
                raise ValueError("Invalid resize ratio")

            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)


            # if os.path.exists(traj_path):
            #     pose_np_gt = loaded_poses[self.idx]
            #     T_frame_to_model = o3c.Tensor(pose_np_gt)
            #     if self.config.add_noise:
            #         xyz_noise, rpy_noise = general_utils.get_noise(
            #             max_xyz_noise=0.1, max_angle_noise_deg=3.)
            #         # xyz_noise = np.array([0.0, 0.0, 0.0])
            #         # rpy_noise = np.array([0.0, 0.0, 0.0])
            #         pose_flat_deg_np_gt = general_utils.extract_xyz_rpw(
            #             pose_np_gt)
            #         T_frame_to_model = general_utils.set_noise(
            #             xyz_noise, rpy_noise, T_frame_to_model,
            #             pose_flat_deg_np_gt, avg_noise_deg)

            if self.idx > 0:
                result = self.model.track_frame_to_model(
                    input_frame,
                    raycast_frame,
                    float(self.scale_slider.int_value),
                    self.max_slider.double_value,
                )
                T_frame_to_model = T_frame_to_model @ result.transformation
                ################# For logging. (avg_fixed_minus_gt_deg)
                # if self.config.add_noise:
                #     recovered_pose_np = T_frame_to_model.cpu().numpy()
                #     recovered_pose_flat_deg_np = general_utils.extract_xyz_rpw(
                #         recovered_pose_np)
                #     fixed_minus_gt_deg = np.abs(recovered_pose_flat_deg_np -
                #                                 pose_flat_deg_np_gt)
                #     avg_fixed_minus_gt_deg += fixed_minus_gt_deg
                ##################

            self.poses.append(T_frame_to_model.cpu().numpy())
            self.model.update_frame_pose(self.idx, T_frame_to_model)
            self.model.integrate(input_frame,
                                 float(self.scale_slider.int_value),
                                 self.max_slider.double_value,
                                 self.trunc_multiplier_slider.double_value)
            self.model.synthesize_model_frame(
                raycast_frame, float(self.scale_slider.int_value),
                self.config.depth_min, self.max_slider.double_value,
                self.trunc_multiplier_slider.double_value,
                self.raycast_box.checked)
            #########################
            if self.idx % self.cfg.stride == 0:
                rgb_tensor, depth_tensor, intrinsics, *_ = self.dataset[self.idx]  # resized
                self.intrinsics = intrinsics.cpu().numpy()
                depth_tensor = depth_tensor[..., 0]  # (H, W)
                depth_array = depth_tensor.cpu().numpy()
                rgb_np = rgb_tensor.cpu().numpy()  # (H, W, 3)
                rgb_np = (rgb_np).astype(np.uint8)  # (H, W, 3)
                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)  # (H, W, 3)
                # Assert that bgr_np and depth_array are of the same shape.
                assert bgr_np.shape[:2] == depth_array.shape, (
                    f"Shape mismatch: bgr{bgr_np.shape[:2]} vs depth{depth_array.shape}")

                camera_pose_ = T_frame_to_model.cpu().numpy()  # (4, 4)

                self.core_logic(bgr_np, depth_array, camera_pose=camera_pose_)

            if (self.idx % self.interval_slider.int_value == 0 and
                    self.update_box.checked) \
                    or (self.idx == 3) \
                    or (self.idx == n_files - 1):
                pcd = self.model.voxel_grid.extract_point_cloud(
                    3.0, self.est_point_count_slider.int_value).to(
                        o3d.core.Device('CPU:0'))
                self.is_scene_updated = True
            else:
                self.is_scene_updated = False

            frustum = o3d.geometry.LineSet.create_camera_visualization(
                color.columns, color.rows, intrinsic.numpy(),
                np.linalg.inv(T_frame_to_model.cpu().numpy()), 0.2)
            frustum.paint_uniform_color([0.961, 0.475, 0.000])

            # Output FPS
            if (self.idx % fps_interval_len == 0):
                end = time.time()
                elapsed = end - start
                start = time.time()
                self.output_fps.text = 'FPS: {:.3f}'.format(fps_interval_len /
                                                            elapsed)

            # Output info
            info = 'Frame {}/{}\n\n'.format(self.idx, n_files)
            info += 'Transformation:\n{}\n'.format(
                np.array2string(T_frame_to_model.numpy(),
                                precision=3,
                                max_line_width=40,
                                suppress_small=True))
            info += 'Active voxel blocks: {}/{}\n'.format(
                self.model.voxel_grid.hashmap().size(),
                self.model.voxel_grid.hashmap().capacity())
            info += 'Surface points: {}/{}\n'.format(
                0 if pcd is None else pcd.point.positions.shape[0],
                self.est_point_count_slider.int_value)

            self.output_info.text = info

            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_render(
                    input_frame.get_data_as_image('depth'),
                    input_frame.get_data_as_image('color'),
                    raycast_frame.get_data_as_image('depth'),
                    raycast_frame.get_data_as_image('color'), pcd, frustum))

            self.idx += 1
            self.is_done = self.is_done | (self.idx >= n_files)

        time.sleep(0.5)
#         if self.config.add_noise:
#             """
# avg_fixed_minus_gt_deg:  [0.04 0.06 0.03 0.27 0.16 0.23]
# avg_noise_deg:  [0. 0. 0. 0. 0. 0.]
#             """
#             avg_fixed_minus_gt_deg /= n_files
#             avg_noise_deg /= n_files
#             print("-----------------")
#             print("avg_fixed_minus_gt_deg: ", np.round(avg_fixed_minus_gt_deg,
#                                                        2))
#             print("avg_noise_deg: ", np.round(avg_noise_deg, 2))
#             print("-----------------")

    def core_logic(self, bgr_np: np.ndarray, depth_array: np.ndarray,
                   depth_builtin_time = None,
                   camera_pose: Optional[np.ndarray] = None):
        color_path = None
        #### 1. frame 처리
        first_start_time = time.time()
        self.frame_idx += 1
        if self.intrinsics is None:
            return
        if camera_pose is None:
            agent_pose = self._get_pose_data(depth_builtin_time)
            if agent_pose is None:
                return
            camera_pose = agent_pose @ self.extrinsic
        self.tracker.curr_frame_idx = self.frame_idx
        self.counter += 1
        self.orr.set_time_sequence("frame", self.frame_idx)

        # Read info about current frame from dataset
        # color image
        # color and depth tensors, and camera instrinsics matrix
        """
        color_path -> image_original_pil (PIL) # 필요 없음
        color_path -> image_rgb (cv2)
            - run_detections가 True일 때, 아래 image_rgb 를 대체
        color_tensor: (680, 1200, 3) -> image_rgb: (680, 1200, 3) # resized
        depth_tensor: (680, 1200, 1) -> depth_array: (680, 1200) # resized
        intrinsics: (4, 4)  # resize 가 들어간 것
        """

        # det_exp_vis_path:Datasets/Replica/room0/exps/s_detections_stride10/vis
        # color_path: Datasets/Replica/room0/results/frame000000.jpg
        # vis_save_path_for_vlm
        # room0/exps/s_detections_stride10/vis/frame000000annotated_for_vlm.jpg
        # CHECK: 내가 수정했음
        vis_save_path_for_vlm = get_vlm_annotated_image_path(
            self.det_exp_vis_path, color_path, frame_idx=self.frame_idx)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(
            self.det_exp_vis_path,
            color_path,
            frame_idx=self.frame_idx,
            w_edges=True)
        if self.run_detections:
            ##### 1.1. [시작] RGBD에서 instance segmentation 진행
            results = self.detection_model.predict(
                bgr_np, conf=self.cfg.mask_conf_threshold, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(
                int)  # (N,)

            # detection_class_labels = [ "sofa chair 0", ... ]
            detection_class_labels = [
                f"{self.obj_classes.get_classes_arr()[class_id]} {class_idx}"
                for class_idx, class_id in enumerate(detection_class_ids)
            ]
            object_number = len(detection_class_ids)
            # 원본 size 기준으로 xyxy 가 나온다는 것을 확인함
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()  # (N, 4)

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                # segmentation
                sam_out = self.sam_predictor.predict(bgr_np,
                                                     bboxes=xyxy_tensor,
                                                     verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()  # (N, H, W)
            else:
                # color_tensor: (H, W, 3)
                H, W, _ = bgr_np.shape
                color_tensor = np.zeros((H, W, 3), dtype=np.uint8)
                masks_np = np.empty((0, *color_tensor.shape[:2]),
                                    dtype=np.float64)
            # Create a detections object that we will save later.
            # TODO: check -> xyxy_np.copy()
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            ##### 1.1. [끝] RGBD에서 instance segmentation 진행

            # Make the edges
            # detection_class_labels: ["sofa chair 0", ...]
            # det_exp_path:
            # Datasets/Replica/room0/exps/s_detections_stride10
            # self.det_exp_vis_path:
            # Datasets/Replica/room0/exps/s_detections_stride10/vis
            # color_path
            # Datasets/Replica/room0/results/frame000000.jpg
            # self.cfg.make_edges: True
            """ make_vlm_edges_and_captions
            labels: ["sofa chair 0", ...]
            edges: [("1", "on top of", "2"), ("3", "under", "2"), ...]
                - 현 코드로는 []
            edge_image: <PIL.Image.Image> node와 edge가 그려진 이미지
                - 현 코드로는 node만 그려진다.
            captions: 
[
    {"id": "1", "name": "object1", "caption": "concise description of object1"},
    {"id": "2", "name": "object2", "caption": "concise description of object2"}
]
                - 현 코드로는 []
            로직
                - segmentation이 끝난 이미지 1장을 VLM에 넣어서
                  - 바로 edge 정보를 구한다.
                  - 바로 node의 caption 정보를 구한다.
                    - 이미 segmentation label이 있지만, 
                        - 부정확할 수 있으므로 VLM을 통해 caption을 구한다고 한다.
            """

            # image: np.zeros (H, W, 3)
            # self.obj_classes: ObjectClasses
            # detection_class_labels = [ "sofa chair 0", ... ]
            # det_exp_vis_path: Datasets/Replica/room0/exps/s_detections_stride10/vis
            # color_path: None
            #

            # labels: detection_class_labels 가 필터링된 것: ["sofa chair 0", ...]
            # edges = [] / edge_image = None / captions = []
            """ make_vlm_edges_and_captions
  - IoU가 80% 이상 겹치면, 신뢰도가 낮은 객체를 제거
  - bg_classes 클래스 제거

  - 위 결과를 저장
    - vis_save_path_for_vlm

            """
            ##### 1.2. [시작] 후처리 후, "프레임 w 노드" 그림을 그리고,
            # VLM에 통과시켜 edge와 개별 물체 caption 정보를 구한다.
            """
            captions
[
    {"id": "1", "name": "object1", "caption": "concise description of object1"},
    {"id": "2", "name": "object2", "caption": "concise description of object2"}
]
            """
            edges = []
            captions = []
            # labels, edges, _, captions = make_vlm_edges_and_captions(
            #     bgr_np,
            #     curr_det,
            #     self.obj_classes,
            #     detection_class_labels,
            #     self.det_exp_vis_path,
            #     color_path,
            #     self.cfg.make_edges,
            #     self.openai_client,
            #     self.frame_idx,
            #     save_result=True)
            # TODO: 더비겅이 완료된 후엔, save_result = False로 해야함.
            ##### 1.2. [끝]
            """
        image_crops: List[Image.Image]
            - 잘라낸 이미지들의 리스트 (길이: N)
        image_feats: np.ndarray
            - 잘라낸 이미지들의 CLIP feature: shape (N, 512)
        text_feats: List
            - 빈 리스트
            """
            if curr_det.xyxy.shape[0] == 0:
                print("No detections found in the image")
                self.prev_adjusted_pose = camera_pose
                return
            ##### 1.3. [시작] 개별 objects에 대한 CLIP feature를 계산
            # image_rgb: (H, W, 3) 원본 사이즈
            rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
            image_crops, image_feats, text_feats = compute_clip_features_batched(
                rgb_np, curr_det, self.clip_model,
                self.clip_preprocess, self.clip_tokenizer,
                self.obj_classes.get_classes_arr(), self.cfg.device)
            ##### 1.3. [끝]

            # increment total object detections
            self.tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection
                "xyxy": curr_det.xyxy,  # (34, 4)
                "confidence": curr_det.confidence,  # (34,)
                "class_id": curr_det.class_id,  # (34,)
                "mask": curr_det.mask,  # (34, 680, 1200)
                "classes": self.obj_classes.get_classes_arr(
                ),  # len = 200, "alarm clock"
                "image_crops": image_crops,  # len = 34, <PIL.Image.Image>
                "image_feats": image_feats,  # (34, 1024)
                "text_feats": text_feats,  # len = 0 # 아마?
                "detection_class_labels":
                    detection_class_labels,  # len = 34, "sofa chair 0"
                # len = 19, "sofa chair 0" -> detection_class_labels을 필터링한 것
                # TODO: 이게 왜 필요하지?
                "labels": detection_class_labels,
                # TODO: 이게 왜 필요하지?
                "edges": edges,  # len = 0
                "captions": captions,  # len = 0
            }
            raw_grounded_obs = results

            # save the detections if needed
            # important
            ##### 1.4. [시작] frame 내 결과 3장 저장하기
            if self.cfg.save_detections:
                # self.det_exp_vis_path:
                # Datasets/Replica/room0/exps/s_detections_stride10/vis
                # color_path
                # Datasets/Replica/room0/results/frame000000.jpg
                # vis_save_path:
                # Datasets/Replica/room0/exps/s_detections_stride10/vis/frame000000.jpg
                """ 3장 그림 그려서 저장하는 과정임
                vis_save_path: bounding box와 mask가 모두 그려진 이미지

                """
                frame_name = f"{self.frame_idx:06}"
                vis_save_path = (self.det_exp_vis_path /
                                 frame_name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(
                    bgr_np, curr_det, self.obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)
                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255,
                                                cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb,
                                               cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(
                    depth_image_rgb, curr_det,
                    self.obj_classes.get_classes_arr())
                cv2.imwrite(
                    str(vis_save_path).replace(".jpg", "_depth.jpg"),
                    annotated_depth_image)
                cv2.imwrite(
                    str(vis_save_path).replace(".jpg", "_depth_only.jpg"),
                    depth_image_rgb)
                save_detection_results(
                    self.det_exp_pkl_path / vis_save_path.stem, results)
            ##### 1.4. [끝] frame 내 결과 3장 저장하기
        """
카메라의 내재적(intrinsic) 및 외재적(extrinsic) 파라미터를 로깅하는 역할
특히, 카메라의 현재 위치와 자세(orientation)를 기록하고,
    "이전 프레임의 카메라 위치"와 "현재 프레임의 카메라 위치"를 연결하는 경로를 시각적으로 나타냄
        """
        self.prev_adjusted_pose = orr_log_camera(torch.from_numpy(self.intrinsics), camera_pose,
                                                 self.prev_adjusted_pose,
                                                 self.cfg.image_width,
                                                 self.cfg.image_height,
                                                 self.frame_idx)

        orr_log_rgb_image(color_path)
        # orr_log_annotated_image(color_path, self.det_exp_vis_path)
        depth_tensor = torch.tensor(depth_array)  # check
        orr_log_depth_image(depth_tensor)  # resized
        orr_log_vlm_image(vis_save_path_for_vlm)
        orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")
        """
            results = {
                # add new uuid for each detection
                "xyxy": curr_det.xyxy,  # (34, 4)
                "confidence": curr_det.confidence,  # (34,)
                "class_id": curr_det.class_id,  # (34,)
                "mask": curr_det.mask,  # (34, 680, 1200)
                "classes":
                    self.obj_classes.get_classes_arr(),  # len = 200, "alarm clock"
                "image_crops": image_crops,  # len = 34, <PIL.Image.Image>
                "image_feats": image_feats,  # (34, 1024)
                "text_feats": text_feats,  # len = 0 # 아마?
                "detection_class_labels":
                    detection_class_labels,  # len = 34, "sofa chair 0"
                "labels": labels,  # len = 19, "sofa chair 0"
                "edges": edges,  # len = 0
                "captions": captions,  # len = 0
            }
        image_rgb: (H, W, 3)
        """
        # CHECK: 아마 현재는 resize 가 필요 없어 보입니다.
        resized_grounded_obs = resize_gobs(raw_grounded_obs, bgr_np)
        resized_grounded_obs = raw_grounded_obs
        ##### 1.5. [시작]  프레임 내 segmentation 결과 필터링
        """
 **필터링 기준 설정**:
    매우 작거나(25) + 배경을 제거하거나 (skip_bg=True) + 이미지 크기의 50% 이상 물체 제거
        """
        filtered_grounded_obs = filter_gobs(
            resized_grounded_obs,
            rgb_np,
            skip_bg=self.cfg.skip_bg,
            # ["wall", "floor", "ceiling"]
            BG_CLASSES=self.obj_classes.get_bg_classes_arr(),
            mask_area_threshold=self.cfg.mask_area_threshold,  # 25
            max_bbox_area_ratio=self.cfg.max_bbox_area_ratio,  # 0.5
            mask_conf_threshold=None,  # self.cfg.mask_conf_threshold, # 0.25
        )

        grounded_obs = filtered_grounded_obs

        if len(grounded_obs['mask']) == 0:  # no detections in this frame
            return

        # this helps make sure things like 베개 on 소파 are separate objects.
        grounded_obs['mask'] = mask_subtract_contained(grounded_obs['xyxy'],
                                                       grounded_obs['mask'])

        ##### 1.5. [끝]  프레임 내 segmentation 결과 필터링
        """
-----------all shapes of detections_to_obj_pcd_and_bbox inputs:--------
depth_array.shape: (680, 1200)
grounded_obs['mask'].shape: (N, 680, 1200)
self.intrinsics.cpu().numpy()[:3, :3].shape: (3, 3)
image_rgb.shape: (680, 1200, 3)
camera_pose.shape: (4, 4)
        """
        print("first_elapsed_time: ", round(time.time() - first_start_time, 2))
        #### 2. pcd 처리
        second_start_time = time.time()
        ##### 2.1. [시작] 3d pointcloud 만들기
        # obj_pcds_and_bboxes : [ {'pcd': pcd, 'bbox': bbox} , ... ]
        obj_pcds_and_bboxes: List[Dict[str, Any]] = measure_time(
            detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=grounded_obs['mask'],
            cam_K=self.intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=rgb_np,
            trans_pose=camera_pose,
            min_points_threshold=self.cfg.min_points_threshold,  # 16
            # overlap # "iou", "giou", "overlap"
            spatial_sim_type=self.cfg.spatial_sim_type,  # overlap
            obj_pcd_max_points=self.cfg.obj_pcd_max_points,  # 5000
            device=self.cfg.device,
        )
        ##### 2.1. [끝] 3d pointcloud 만들기
        for obj in obj_pcds_and_bboxes:
            if obj:
                # obj: {'pcd': pcd, 'bbox': bbox}
                """
                포인트 클라우드를 voxel (0.01m)로 다운샘플링 하고,
                dbscan clustering 기반으로
                    노이즈를 제거하여 클러스터링을 통해 중요한 포인트만 남기는 것
                    0.1m 거리 내의 cluster 을 모으는데, 최소 10개 포인트가 있어야 한다.
                """
                ##### 2.2. [시작] pointclouds 필터링
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=self.
                    cfg["downsample_voxel_size"],  # 0.01
                    dbscan_remove_noise=self.cfg["dbscan_remove_noise"],  # True
                    dbscan_eps=self.cfg["dbscan_eps"],  # 0.1
                    dbscan_min_points=self.cfg["dbscan_min_points"],  # 10
                )
                # TODO: 중복된 bounding box 계산 1번으로 줄이기 ?
                # point cloud를 filtering 했으니, bounding box를 다시 계산
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=self.cfg['spatial_sim_type'],  # overlap
                    pcd=obj["pcd"],
                )
                ##### 2.2. [끝] pointclouds 필터링
        """
        obj_pcds_and_bboxes : [ {'pcd': pcd, 'bbox': bbox} , ... ]
            grounded_obs = {
                # add new uuid for each detection
                "xyxy": curr_det.xyxy,  # (34, 4)
                "confidence": curr_det.confidence,  # (34,)
                ...
            }
        # color_path: Datasets/Replica/room0/results/frame000000.jpg
        """
        ##### 2.3. [시작] frame 결과와 frame 3차원 결과 융합
        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, grounded_obs, color_path, self.obj_classes,
            self.frame_idx)
        if len(detection_list) == 0:  # no detections, skip
            return
        ##### 2.3. [끝] frame 결과와 frame 3차원 결과 융합
        print("second_elapsed_time: ", round(time.time() - second_start_time, 2))
        #### 3. 기존 object pcd와 융합
        third_start_time = time.time()
        ##### 3.1. [시작] 기존 object들과 융합하기
        # 아무것도 없었으면, 그냥 추가
        if len(self.objects) == 0:
            self.objects.extend(detection_list)
            self.tracker.increment_total_objects(len(detection_list))
            # TODO: return이 맞나?
            return

        ### compute similarities and then merge
        # spatial_sim : (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 중첩 비율
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=self.cfg['spatial_sim_type'],  # overlap
            detection_list=detection_list,
            objects=self.objects,
            # downsample_voxel_size: pointcloud 거리 기준 (0.01m)
            downsample_voxel_size=self.cfg['downsample_voxel_size'])  # 0.01
        # visual_sim :  (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 코싸인 유사도
        visual_sim = compute_visual_similarities(detection_list, self.objects)

        # match_method = "sim_sum" # "sep_thresh", "sim_sum"
        # "sim_sum": 단순하게 두개를 그냥 더함
        # agg_sim : (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 유사도
        agg_sim = aggregate_similarities(
            match_method=self.cfg['match_method'],
            phys_bias=self.cfg['phys_bias'],  # 0.0
            spatial_sim=spatial_sim,
            visual_sim=visual_sim)

        # Perform matching of detections to existing self.objects .
        # match_indices: 길이는 "새 검지 개수"
        match_indices: List[Optional[int]] = match_detections_to_objects(
            agg_sim=agg_sim,
            detection_threshold=self.cfg['sim_threshold']  # 1.2
        )

        # 병합 후, downsample 진행하고, dbscan 으로 노이즈 잘라냅니다.
        self.objects = merge_obj_matches(
            detection_list=detection_list,
            objects=self.objects,
            match_indices=match_indices,
            downsample_voxel_size=self.cfg['downsample_voxel_size'],  # 0.01
            dbscan_remove_noise=self.cfg['dbscan_remove_noise'],  # True
            dbscan_eps=self.cfg['dbscan_eps'],  # 0.1
            dbscan_min_points=self.cfg['dbscan_min_points'],  # 10
            spatial_sim_type=self.cfg['spatial_sim_type'],  # overlap
            device=self.cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )

        # fix the class names for self.objects
        # they should be the most popular name, not the first name .
        for idx, obj in enumerate(self.objects):
            temp_class_name = obj["class_name"]  # "sofa chair"
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(
                1)[0][0]
            most_common_class_name = self.obj_classes.get_classes_arr(
            )[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        ##### 3.1. [끝] 기존 object들과 융합하기

        ##### 3.2. [시작] edge 계산하기
        if self.cfg["make_edges"]:
            self.map_edges = process_edges(match_indices, grounded_obs,
                                           len(self.objects), self.objects,
                                           self.map_edges, self.frame_idx)
            # Clean up outlier edges
            edges_to_delete = []
            for curr_map_edge in self.map_edges.edges_by_index.values():
                curr_obj1_idx = curr_map_edge.obj1_idx
                curr_obj2_idx = curr_map_edge.obj2_idx
                curr_first_detected = curr_map_edge.first_detected
                curr_num_det = curr_map_edge.num_detections
                if (self.frame_idx - curr_first_detected
                    > 5) and curr_num_det < 2:
                    edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
            for edge in edges_to_delete:
                self.map_edges.delete_edge(edge[0], edge[1])
        ##### 3.2. [끝] edge 계산하기

        #### 4. 주기적 후처리
        print("third_elapsed_time: ", round(time.time() - third_start_time, 2))
        ##### 4.1. [시작] 주기적 "누적 object" 후처리
        fourth_start_time = time.time()
        # Denoising
        if processing_needed(
                # Run DBSCAN every k frame. This operation is heavy
                self.cfg["denoise_interval"],  # 20
                self.cfg["run_denoise_final_frame"],  # True
                self.frame_idx,
        ):
            # TODO: 병합 때 downsample + dbscan 을 했는데, 여기서도 또 한번 하는 이유가 뭘까?
            self.objects = measure_time(denoise_objects)(
                downsample_voxel_size=self.cfg['downsample_voxel_size'],  # 0.01
                dbscan_remove_noise=self.cfg['dbscan_remove_noise'],  # True
                dbscan_eps=self.cfg['dbscan_eps'],  # 0.1
                dbscan_min_points=self.cfg['dbscan_min_points'],  # 10
                spatial_sim_type=self.cfg['spatial_sim_type'],  # overlap
                device=self.cfg['device'],
                objects=self.objects)

        # Filtering
        # 저는 안합니다.
        if processing_needed(
                # Filter objects that have too few associations or are too small.
                self.cfg["filter_interval"],  # 5
                self.cfg["run_filter_final_frame"],  # True
                self.frame_idx,
        ):
            self.objects = filter_objects(
                obj_min_points=self.cfg['obj_min_points'],  # 0
                obj_min_detections=self.cfg['obj_min_detections'],  # 1
                objects=self.objects,
                map_edges=self.map_edges)

        # Merging
        if processing_needed(
                # Merge objects based on geometric and semantic similarity
                self.cfg["merge_interval"],  # 5
                self.cfg["run_merge_final_frame"],  # True
                self.frame_idx,
        ):
            """
             거리 유사도 / 시각적 유사도가 모두 threshold를 넘으면 병합 -> 
                병합 시 , downsampling + dbscan으로 노이즈 제거
            현재까지 만들어진 object 들 자기들끼리 비교
            """
            self.objects, self.map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=self.cfg["merge_overlap_thresh"],  # 0.7
                # Merge only if the visual similarity is larger than this threshold
                merge_visual_sim_thresh=self.
                cfg["merge_visual_sim_thresh"],  # 0.7
                # Merge only if the text similarity is larger than this threshold
                merge_text_sim_thresh=self.cfg["merge_text_sim_thresh"],  # 0.7
                objects=self.objects,
                downsample_voxel_size=self.cfg["downsample_voxel_size"],  # 0.01
                dbscan_remove_noise=self.cfg["dbscan_remove_noise"],  # True
                dbscan_eps=self.cfg["dbscan_eps"],  # 0.1
                dbscan_min_points=self.cfg["dbscan_min_points"],  # 10
                spatial_sim_type=self.cfg["spatial_sim_type"],  # overlap
                device=self.cfg.device,
                do_edges=self.cfg["make_edges"],  # True
                map_edges=self.map_edges)
        orr_log_objs_pcd_and_bbox(self.objects, self.obj_classes)
        orr_log_edges(self.objects, self.map_edges, self.obj_classes)

        ##### 4.1. [끝] 주기적 "누적 object" 후처리

        ##### 4.2. [시작] 매 frame 마다 object 결과 저장 (사용하면 매번 저장)
        if self.cfg.save_objects_all_frames:
            # obj_all_frames_out_path:
            # room0/exps/r_mapping_stride10/saved_obj_all_frames/det_s_detections_stride10/
            #
            # {frame_idx:06d}.pkl.gz
            # obj_min_detections: 1
            """ save_objects_for_frame (아래의 것들을 압축하여 하나의 파일로 저장)
 - `camera_pose`: 조정된 카메라의 위치 정보(`adjusted_pose`).
 - `objects`: 필터링 및 준비된 객체 리스트(`prepared_objects`).
 - `frame_idx`: 현재 프레임 인덱스(`frame_idx`).
 - `num_objects`: 객체의 개수.
 - `color_path`: 프레임에 대한 이미지 경로(`color_path`).
            """
            save_objects_for_frame(self.obj_all_frames_out_path, self.frame_idx,
                                   self.objects, self.cfg.obj_min_detections,
                                   camera_pose, color_path)
        ##### 4.2. [끝] 매 frame 마다 object 결과 저장 (사용하면 매번 저장)

        ##### 4.3. [시작] 주기적으로 3d pcd 결과 저장
        # periodically_save_pcd: False
        # periodically_save_pcd_interval: 10
        if self.cfg.periodically_save_pcd and (
                self.counter % self.cfg.periodically_save_pcd_interval == 0):
            # save the pointcloud
            """
1. **저장할 결과 딕셔너리 준비**
 - `objects`: 객체 리스트로, `to_serializable()` 메서드를 사용해 직렬화 가능한 형태로 변환
 - `cfg`: 실험 설정 정보를 `cfg_to_dict(cfg)` 함수를 통해 딕셔너리로 변환
 - `class_names`: 객체 클래스 이름 배열로, `get_classes_arr()` 메서드를 통해 가져옵니다.
 - `class_colors`: 객체 클래스의 색상 정보를 `get_class_color_dict_by_index()` 메서드를 통해 가져옵니다.
 - `edges`: 객체 간 연결 정보를 직렬화 가능한 형태로 변환하여 포함시키며, `edges`가 제공되지 않은 경우 `None`으로 설정
pcd_save_path = exps/r_mapping_stride10/pcd_r_mapping_stride10.pkl.gz
            """
            save_pointcloud(exp_suffix=self.cfg.exp_suffix,
                            exp_out_path=self.exp_out_path,
                            cfg=self.cfg,
                            objects=self.objects,
                            obj_classes=self.obj_classes,
                            latest_pcd_filepath=self.cfg.latest_pcd_filepath,
                            create_symlink=True)

        self.tracker.increment_total_objects(len(self.objects))
        self.tracker.increment_total_detections(len(detection_list))
        ## 4.3. [끝] 주기적으로 3d pcd 결과 저장
        print("fourth_elapsed_time: ", round(time.time() - fourth_start_time, 2))
        #################

    def wrap_up(self):
        #### 5. wrap_up

        ##### 5.1. [시작] 각 물체마다, LLM 사옹해서 caption 여러개를 하나로 합치기
        for object in self.objects:
            """
            captions
[
    {"id": "1", "name": "object1", "caption": "concise description of object1"},
    {"id": "1", "name": "object1", "caption": "concise 2 description of object1"}
]
            """
            obj_captions = object['captions'][:20]  # 첫 20개만 사용
            # LLM 사옹해서 caption 합치기
            consolidated_caption = consolidate_captions(self.openai_client,
                                                        obj_captions)
            object['consolidated_caption'] = consolidated_caption
        ##### 5.1. [끝] 각 물체마다, LLM 사옹해서 caption 여러개를 하나로 합치기

        # exp_suffix: r_mapping_stride10
        # self.exp_out_path: exps/r_mapping_stride10/
        # rerun_file_path:
        #   exps/r_mapping_stride10/rerun_r_mapping_stride10.rrd
        # PRINT 문만 띄어주는 역할
        ##### 5.2. [시작] rerun 결과 저장하기
        # handle_rerun_saving(
        #     self.cfg.use_rerun,
        #     self.cfg.save_rerun,  # True
        #     self.cfg.exp_suffix,
        #     self.exp_out_path)
        ##### 5.2. [끝]

        ##### 5.3. [시작] save_pcd / object.json , edge.json (save_json)
        if self.cfg.save_pcd:  # True
            # exp_suffix: rerun_r_mapping_stride10
            # self.exp_out_path: exps/r_mapping_stride10/
            """
1. **저장할 결과 딕셔너리 준비**
 - `objects`: 
    객체 리스트로, `to_serializable()` 메서드를 사용해 직렬화 가능한 형태로 변환
 - `cfg`: 
    실험 설정 정보를 `cfg_to_dict(cfg)` 함수를 통해 딕셔너리로 변환
 - `class_names`: 
    객체 클래스 이름 배열로, `get_classes_arr()` 메서드를 통해 가져옴
 - `class_colors`: 
    객체 클래스의 색상 정보를 `get_class_color_dict_by_index()` 메서드를 통해 가져옴
 - `edges`: 
    객체 간 연결 정보를 직렬화 가능한 형태로 변환하여 포함시키며, `edges`가 제공되지 않은 경우 `None`으로 설정

pcd_save_path = exps/r_mapping_stride10/pcd_r_mapping_stride10.pkl.gz
            """
            save_pointcloud(
                exp_suffix=self.cfg.exp_suffix,
                exp_out_path=self.exp_out_path,
                cfg=self.cfg,
                objects=self.objects,
                obj_classes=self.obj_classes,
                # ./latest_pcd_save
                latest_pcd_filepath=self.cfg.latest_pcd_filepath,
                create_symlink=True,
                edges=self.map_edges)

        if self.cfg.save_json:
            """
                {"object_1": {
                    "id": 1,
                    "object_tag": "stool",
                    "object_caption": "",
                    "bbox_extent": [ 0.65, 0.68, 0.47 ],
                    "bbox_center": [ 1.8, 0.66, -1.23 ],
                    "bbox_volume": 0.21 }, ...
            """
            # /room0/exps/r_mapping_stride10/obj_json_r_mapping_stride10.json
            save_obj_json(exp_suffix=self.cfg.exp_suffix,
                          exp_out_path=self.exp_out_path,
                          objects=self.objects)

            # /room0/exps/r_mapping_stride10/edge_json_r_mapping_stride10.json
            save_edge_json(exp_suffix=self.cfg.exp_suffix,
                           exp_out_path=self.exp_out_path,
                           objects=self.objects,
                           edges=self.map_edges)
        ##### 5.3. [끝]

        ##### 5.4.[시작] 그 외(모든 프레임이 저장될 때 metadata, detection 결과 비디오) 저장하기
        if self.cfg.save_objects_all_frames:
            # obj_all_frames_out_path:
            # room0/exps/r_mapping_stride10/saved_obj_all_frames/det_s_detections_stride10/
            save_meta_path = self.obj_all_frames_out_path / f"meta.pkl.gz"
            with gzip.open(save_meta_path, "wb") as f:
                pickle.dump(
                    {
                        'cfg':
                            self.cfg,
                        'class_names':
                            self.obj_classes.get_classes_arr(),
                        'class_colors':
                            self.obj_classes.get_class_color_dict_by_index(),
                    }, f)

        if self.run_detections:
            if self.cfg.save_video:
                # "room0/exps/s_detections_stride10/vis_video.mp4"
                save_video_detections(self.det_exp_path)
        ##### 5.4. [끝]


@hydra.main(version_base=None,
            config_path="../hydra_configs/",
            config_name="realsense_realtime_mapping")
def main(cfg: DictConfig):
    parser = ConfigParser()
    """ 얘가 default_config.yml을 읽어서 config를 만들어줌
`--config` 인자를 추가할 때 `is_config_file=True` 옵션을 사용했기 때문에 해당 파일을 자동으로 읽어오는 것
이 동작은 일반적인 `argparse` 라이브러리에서는 제공되지 않으며, 확장 라이브러리인 `configargparse`에서 제공하는 기능

### 1. **`configargparse`의 동작 원리**
- `configargparse`는 `argparse`와 호환되지만, 설정 파일을 직접 읽고 처리할 수 있는 기능을 제공
- `parser.add(..., is_config_file=True)`는 `configargparse`를 사용하여 설정 파일을 지정할 수 있는 인자를 정의
- 실행 파일과 같은 경로에 `default_config.yml`이 있는 경우, `configargparse`는 해당 파일을 자동으로 찾고 읽어들임

### 2. **자동으로 설정 파일을 읽는 원리**
`configargparse`는 기본적으로 설정 파일을 자동으로 로드하는 동작을 갖고 있으며, 다음과 같은 원리로 동작합니다:

#### 2.1. `is_config_file=True` 옵션
- `is_config_file=True`를 사용하면 `configargparse`는 해당 인자를 설정 파일로 인식합니다. 
    - 즉, 이 인자를 통해 설정 파일의 경로를 받아들이는 역할을 합니다.
- `--config` 인자를 명령줄에서 명시하지 않더라도, 
    - `configargparse`는 실행 파일의 경로를 기반으로 `default_config.yml`과 같은 일반적인 파일 이름을 자동으로 찾습니다.

#### 2.2. 내부 파일 검색 및 로드
- `configargparse`는 `ArgumentParser`가 생성될 때, 
    - `default_config_files`와 현재 경로에 있는 파일들을 함께 검색하여 설정 파일이 있는지 확인합니다.
- 파일 이름이 `default_config.yml`과 같이 일반적으로 많이 쓰이는 이름인 경우, 자동으로 이를 설정 파일로 인식하여 읽어들이는 기능을 제공
    """
    # /home/hsb/PycharmProjects/Open3D/examples/python/t_reconstruction_system/config.py
    # 위 경로에서 기본 설설정 파일을 "default_config.yml"로 지정해놨음
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
             'reference. It overrides the default config file, but will be '
             'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
                    'Default dataset may be selected from the following options: '
                    '[lounge, bedroom, jack_jack]',
               default='lounge')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='output.npz')
    parser.add('--add_noise',
               type=bool,
               default=True,
               help='Add noise to the poses')
    config = parser.get_config()

    if config.path_dataset == '':
        config = get_default_dataset(config)

    # Extract RGB-D frames and intrinsic from bag file.
    if config.path_dataset.endswith(".bag"):
        assert os.path.isfile(
            config.path_dataset), f"File {config.path_dataset} not found."
        print("Extracting frames from RGBD video file")
        config.path_dataset, config.path_intrinsic, config.depth_scale = extract_rgbd_frames(
            config.path_dataset)

    parser = argparse.ArgumentParser(
        description='Open3D Reconstruction System')
    parser.add_argument('--realsense_idx',
                        type=int,
                        default=0,
                        help="Robot ID for the RealtimeHumanSegmenterNode Node")
    parser.add_argument('--min_points_threshold',
                        type=int,
                        default=16,
                        help="Minimum number of points required for an object")
    parser.add_argument('--conf',
                        type=float,
                        default=0.2,
                        help="Confidence threshold for detection")
    parser.add_argument('--resize_ratio',
                        type=float,
                        default=1.,
                        help="Resize ratio for the image")
    parser.add_argument(
        '--obj_pcd_max_points',
        type=int,
        default=5000,
        help="Maximum number of points in an object's point cloud")
    args = parser.parse_args()

    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = ReconstructionWindow(config, mono, cfg, args)
    app.run()

if __name__ == '__main__':
    main()


