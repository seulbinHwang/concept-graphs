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

from conceptgraph.utils.config import ConfigParser

import os
import numpy as np
import threading
import time
from conceptgraph.utils.common import load_rgbd_file_names, save_poses, load_intrinsic, extract_trianglemesh, get_default_dataset, extract_rgbd_frames
from conceptgraph.utils import general_utils


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable


class ReconstructionWindow:

    def __init__(self, config, font_id):
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
            print('Saving model to {}...'.format(config.path_npz))
            self.model.voxel_grid.save(config.path_npz)
            print('Finished.')

            mesh_fname = '.'.join(config.path_npz.split('.')[:-1]) + '.ply'
            print('Extracting and saving mesh to {}...'.format(mesh_fname))
            mesh = extract_trianglemesh(self.model.voxel_grid, config,
                                        mesh_fname)
            print('Finished.')

            log_fname = '.'.join(config.path_npz.split('.')[:-1]) + '.log'
            print('Saving trajectory to {}...'.format(log_fname))
            save_poses(log_fname, self.poses)
            print('Finished.')

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
                                     config.depth_min,
                                     self.max_slider.double_value).to_legacy())
        self.input_color_image.update_image(color_ref.to_legacy())

        self.raycast_depth_image.update_image(
            depth_ref.colorize_depth(float(self.scale_slider.int_value),
                                     config.depth_min,
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
                float(self.scale_slider.int_value), config.depth_min,
                self.max_slider.double_value).to_legacy())
        self.input_color_image.update_image(input_color.to_legacy())

        self.raycast_depth_image.update_image(
            raycast_depth.colorize_depth(
                float(self.scale_slider.int_value), config.depth_min,
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

        n_files = len(color_file_names)
        traj_path = os.path.join(self.config.path_dataset, "traj.txt")
        if os.path.exists(traj_path):
            loaded_poses = general_utils.load_poses(traj_path, n_files)
            print("Loaded poses from {}".format(traj_path))
        else:
            T_frame_to_model = o3c.Tensor(np.identity(4))
        device = o3d.core.Device(config.device)
        depth_ref = o3d.t.io.read_image(depth_file_names[0])
        color_ref = o3d.t.io.read_image(color_file_names[0])
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

            depth = o3d.t.io.read_image(depth_file_names[self.idx]).to(device)
            color = o3d.t.io.read_image(color_file_names[self.idx]).to(device)

            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)
            if os.path.exists(traj_path):
                pose_np_gt = loaded_poses[self.idx]
                T_frame_to_model = o3c.Tensor(pose_np_gt)
                if config.add_noise:
                    xyz_noise, rpy_noise = general_utils.get_noise(
                        max_xyz_noise=0.1, max_angle_noise_deg=3.)
                    # xyz_noise = np.array([0.0, 0.0, 0.0])
                    # rpy_noise = np.array([0.0, 0.0, 0.0])
                    pose_flat_deg_np_gt = general_utils.extract_xyz_rpw(
                        pose_np_gt)
                    T_frame_to_model = general_utils.set_noise(
                        xyz_noise, rpy_noise, T_frame_to_model,
                        pose_flat_deg_np_gt, avg_noise_deg)

            if self.idx > 0:
                self.model.update_frame_pose(self.idx, T_frame_to_model)
                self.model.synthesize_model_frame(
                    raycast_frame, float(self.scale_slider.int_value),
                    config.depth_min, self.max_slider.double_value,
                    self.trunc_multiplier_slider.double_value,
                    self.raycast_box.checked)
                result = self.model.track_frame_to_model(
                    input_frame,
                    raycast_frame,
                    float(self.scale_slider.int_value),
                    self.max_slider.double_value,
                )
                T_frame_to_model = T_frame_to_model @ result.transformation
                ################# For logging. (avg_fixed_minus_gt_deg)
                if config.add_noise:
                    recovered_pose_np = T_frame_to_model.cpu().numpy()
                    recovered_pose_flat_deg_np = general_utils.extract_xyz_rpw(
                        recovered_pose_np)
                    fixed_minus_gt_deg = np.abs(recovered_pose_flat_deg_np -
                                                pose_flat_deg_np_gt)
                    avg_fixed_minus_gt_deg += fixed_minus_gt_deg
                ##################

            self.poses.append(T_frame_to_model.cpu().numpy())
            self.model.update_frame_pose(self.idx, T_frame_to_model)
            self.model.integrate(input_frame,
                                 float(self.scale_slider.int_value),
                                 self.max_slider.double_value,
                                 self.trunc_multiplier_slider.double_value)

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
        if config.add_noise:
            """
avg_fixed_minus_gt_deg:  [0.04 0.06 0.03 0.27 0.16 0.23]
avg_noise_deg:  [0. 0. 0. 0. 0. 0.]
            """
            avg_fixed_minus_gt_deg /= n_files
            avg_noise_deg /= n_files
            print("-----------------")
            print("avg_fixed_minus_gt_deg: ", np.round(avg_fixed_minus_gt_deg,
                                                       2))
            print("avg_noise_deg: ", np.round(avg_noise_deg, 2))
            print("-----------------")


if __name__ == '__main__':
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

    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = ReconstructionWindow(config, mono)
    app.run()

