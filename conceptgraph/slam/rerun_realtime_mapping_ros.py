'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import os
import copy
import uuid
import time
from pathlib import Path
import pickle
import gzip
from typing import List, Dict, Any, Optional

# Third-party imports
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from tqdm import trange
# from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import Counter
from builtin_interfaces.msg import Time
# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, orr_log_annotated_image, orr_log_camera, orr_log_depth_image,
    orr_log_edges, orr_log_objs_pcd_and_bbox, orr_log_rgb_image,
    orr_log_vlm_image)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_obj_rel_from_image_gpt4v, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, find_existing_image_path, get_det_out_path, get_exp_out_path,
    get_vlm_annotated_image_path, handle_rerun_saving, load_saved_detections,
    load_saved_hydra_json_config, make_vlm_edges_and_captions, measure_time,
    save_detection_results, save_edge_json, save_hydra_config, save_obj_json,
    save_objects_for_frame, save_pointcloud, should_exit_early,
    vis_render_image)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (OnlineObjectRenderer,
                                    save_video_from_frames,
                                    vis_result_fast_on_depth,
                                    vis_result_for_vlm, vis_result_fast,
                                    save_video_detections)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs, filter_objects, get_bounding_box, init_process_pcd,
    make_detection_list_from_pcd_and_gobs, denoise_objects, merge_objects,
    detections_to_obj_pcd_and_bbox, prepare_objects_save_vis, process_cfg,
    process_edges, process_pcd, processing_needed, resize_gobs)
from conceptgraph.slam.mapping import (compute_spatial_similarities,
                                       compute_visual_similarities,
                                       aggregate_similarities,
                                       match_detections_to_objects,
                                       merge_obj_matches)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections
"""
python -m conceptgraph.slam.rerun_realtime_mapping
"""

# Disable torch gradient computation
torch.set_grad_enabled(False)
RUN_OPEN_API = False
RUN_START = False
RUN_MIDDLE = False
RUN_AFTER = False

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import CompressedImage, CameraInfo
# from vision_msgs.msg import BoundingBox2DArray, BoundingBox2D
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Union, Optional
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
import tf_transformations
import argparse
from tf2_ros import (ConnectivityException, ExtrapolationException,
                     LookupException)
from scipy.spatial.transform import Rotation as R
import os
import traceback


class RealtimeHumanSegmenterNode(Node):

    def __init__(self, cfg: DictConfig, args: argparse.Namespace):
        super().__init__('ros2_bridge')
        self.cfg = cfg
        self.args = args
        self._target_frame = "odom"  #"vl"
        self._source_frame = "base_link"

        # tracker : **탐지된 객체**, **병합된 객체** 및 **운영 수**와 같은 여러 상태 정보를 관리
        self.tracker = MappingTracker()
        # 만약 Rerun이 설치되어 있지 않거나, 사용하지 않는 경우, 이 변수는 None입니다.
        self.orr = OptionalReRun()
        self.orr.set_use_rerun(cfg.use_rerun)  # True
        self.orr.init("realtime_mapping")
        self.orr.spawn()

        cfg = process_cfg(cfg)
        self.objects = MapObjectList(device=args.device)
        self.map_edges = MapEdgeMapping(self.objects)

        # output folder for this mapping experiment
        # dataset_root: Datasets
        # scene_id: Replica/room0
        # exp_suffix: r_mapping_stride10
        # self.exp_out_path: Datasets/Replica/room0/exps/r_mapping_stride10
        self.exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id,
                                             cfg.exp_suffix)

        # output folder of the detections experiment to use
        # det_exp_path: Datasets/Replica/room0/exps/s_detections_stride10
        self.det_exp_path = get_exp_out_path(cfg.dataset_root,
                                             cfg.scene_id,
                                             cfg.detections_exp_suffix,
                                             make_dir=False)

        # we need to make sure to use the same classes as the ones used in the detections
        detections_exp_cfg = cfg_to_dict(cfg)
        # obj_classes 에서 person 제외했음
        self.obj_classes = ObjectClasses(
            classes_file_path=detections_exp_cfg['classes_file'],
            bg_classes=detections_exp_cfg['bg_classes'],
            skip_bg=detections_exp_cfg['skip_bg'])

        # if we need to do detections
        # det_exp_path:
        # concept-graphs/Datasets/Replica/room0/exps/s_detections_stride10
        # check_run_detections: s_detections_stride10 폴더가 있으면 실시하지 않음
        self.run_detections = check_run_detections(cfg.force_detection,
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
            self.detection_model = measure_time(YOLO)('yolov8l-world.pt')
            self.sam_predictor = SAM(
                'sam_b.pt')  # SAM('mobile_sam.pt') # UltraLytics SAM
            (self.clip_model, _,
             self.clip_preprocess) = open_clip.create_model_and_transforms(
                 "ViT-H-14", "laion2b_s32b_b79k")
            self.clip_model = self.clip_model.to(cfg.device)
            self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

            # Set the classes for the detection model
            self.detection_model.set_classes(self.obj_classes.get_classes_arr())
            if RUN_OPEN_API:
                self.openai_client = get_openai_client()
            else:
                self.openai_client = None

        else:
            print("\n".join(["NOT Running detections..."] * 10))

        save_hydra_config(cfg, self.exp_out_path)
        save_hydra_config(detections_exp_cfg,
                          self.exp_out_path,
                          is_detection_config=True)

        if cfg.save_objects_all_frames:
            # exp_out_path: Datasets/Replica/room0/exps/r_mapping_stride10
            # obj_all_frames_out_path: room0/exps/r_mapping_stride10/saved_obj_all_frames/det_s_detections_stride10
            self.obj_all_frames_out_path = (self.exp_out_path /
                                            "saved_obj_all_frames" /
                                            f"det_{cfg.detections_exp_suffix}")
            os.makedirs(self.obj_all_frames_out_path, exist_ok=True)

        self.counter = 0
        ###################################
        self.frame_idx = -1
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._frame_idx = 0
        self.extrinsic = self._set_extrinsic()
        # bbox_2d_topic = \
        #     f"realsense{self.args.realsense_idx}/bounding_boxes_2d"
        # self.bounding_boxes_pub = self.create_publisher(BoundingBox2DArray,
        #                                                 bbox_2d_topic, 10)

        self._set_rgbd_info_subscribers()
        self._set_rgbd_subscribers()

    def _set_rgbd_subscribers(self):
        rgb_sub_topic_name = f"realsense{self.args.realsense_idx}/color"
        rgb_sub = message_filters.Subscriber(self, CompressedImage,
                                             rgb_sub_topic_name)
        depth_sub_topic_name = \
            f"realsense{self.args.realsense_idx}/depth"
        depth_sub = message_filters.Subscriber(self, CompressedImage,
                                               depth_sub_topic_name)
        """
        TODO: 큰 문제
        - 부탁드려서, rgb 보내는 것과 depth 보내는 것의 timestamp를 동일하게 해달라고 부탁 
        """
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.02)
        self.ts.registerCallback(self.sync_callback)

    @staticmethod
    def save_cropped_images(image_crops: List[Image.Image], folder_path: str,
                            frame_idx) -> None:
        # 폴더가 존재하지 않으면 생성
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 각각의 이미지들을 폴더에 저장
        for i, cropped_image in enumerate(image_crops):
            # 이미지 파일 이름을 생성 (예: crop_0.png, crop_1.png ...)
            image_path = os.path.join(folder_path, f"[{frame_idx}]crop_{i}.png")

            # 이미지 저장
            cropped_image.save(image_path)

    def _set_rgbd_info_subscribers(self):
        color_camera_info_topic_name = \
            f"realsense{self.args.realsense_idx}/color_camera_info"
        self.rgb_intrinsics = self.rgb_dist_coeffs = None
        self.color_camera_info_sub = self.create_subscription(
            CameraInfo, color_camera_info_topic_name,
            self.color_camera_info_callback, 10)

        depth_camera_info_topic_name = \
            f"realsense{self.args.realsense_idx}/depth_camera_info"
        self.intrinsics = self.depth_dist_coeffs = None
        self.depth_camera_info_sub = self.create_subscription(
            CameraInfo, depth_camera_info_topic_name,
            self.depth_camera_info_callback, 10)

    def _set_extrinsic(self) -> np.ndarray:
        axis_transpose_matrix = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                                          [1., 0., 0., 0.], [0., 0., 0., 1.]])
        if self.args.realsense_idx == 0:  # 로봇 정면 기준 왼쪽 카메라 (+y)
            camera_translation = np.array([0.23547, 0.10567, 0.90784])
            camera_radian_rotation = np.deg2rad(np.array(
                [0., 52., 20.]))  # yaw 회전 후 -> roll -> ptich
            rotation = R.from_euler(
                'xyz', camera_radian_rotation
            )  # same as yaw_matrix @ pitch_matrix @ roll_matrix
            rotation_matrix_3 = rotation.as_matrix()
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = rotation_matrix_3
        elif self.args.realsense_idx == 2:  # 로봇 정면 기준 오른쪽 카메라 (-y)
            camera_translation = np.array([0.23547, -0.10567, 0.90784])
            camera_degree_rotation = np.deg2rad(np.array(
                [0., 52., -20.]))  # yaw 회전 후 -> roll -> ptich
            rotation = R.from_euler('xyz', camera_degree_rotation)
            rotation_matrix_3 = rotation.as_matrix()
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = rotation_matrix_3
        else:
            raise ValueError("Invalid realsense_idx")
        translation_matrix = tf_transformations.translation_matrix(
            camera_translation)
        # IMPORTANT !!! world_coord = camera_pose_wrt_agent @ camera_coord
        (camera_pose_wrt_agent
        ) = translation_matrix @ rotation_matrix @ np.linalg.inv(
            axis_transpose_matrix)
        # camera_pose_wrt_agent shape :
        return camera_pose_wrt_agent

    @staticmethod
    def _transform_stamped_to_matrix(transform: TransformStamped) -> np.ndarray:
        # 평행 이동 벡터 추출
        translation = transform.transform.translation
        trans = np.array([translation.x, translation.y, translation.z])

        # 쿼터니언 추출 및 회전 행렬 생성
        rotation = transform.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        rotation_matrix = tf_transformations.quaternion_matrix(quat)  # 4x4 행렬

        # 4x4 변환 행렬 생성
        translation_matrix = np.identity(4)
        translation_matrix[:, :3] = trans
        # world_coord 기준 좌표 = transform_matrix @ agent_coord 기준 좌표
        # TODO: inv(rotation_matrix) 를 써야하는지 확인해보기
        transform_matrix = translation_matrix @ rotation_matrix
        # transform_matrix: (4, 4)
        return transform_matrix

    def _add_all_true_mask(self, rgb_array: np.ndarray,
                           masks_np: np.ndarray) -> np.ndarray:
        H, W, _ = rgb_array.shape
        all_true_mask = np.expand_dims(np.ones((H, W), dtype=np.uint8), axis=0)
        if (masks_np is None) or (len(masks_np) == 0):
            masks_np = all_true_mask
        else:
            masks_np = np.concatenate([all_true_mask, masks_np], axis=0)
        return masks_np

    def sync_callback(self, rgb_msg: CompressedImage,
                      depth_msg: CompressedImage):
        #### 1. frame 처리
        self.frame_idx += 1

        color_path = None
        if self.intrinsics is None:
            return

        rgb_array, rgb_builtin_time = self.rgb_callback(rgb_msg)
        depth_array, depth_builtin_time = self.depth_callback(depth_msg)
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
                rgb_array, conf=self.cfg.mask_conf_threshold, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(
                int)  # (N,)
            # detection_class_labels = [ "sofa chair 0", ... ]
            detection_class_labels = [
                f"{self.obj_classes.get_classes_arr()[class_id]} {class_idx}"
                for class_idx, class_id in enumerate(detection_class_ids)
            ]
            # 원본 size 기준으로 xyxy 가 나온다는 것을 확인함
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()  # (N, 4)

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                # segmentation
                sam_out = self.sam_predictor.predict(rgb_array,
                                                     bboxes=xyxy_tensor,
                                                     verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()  # (N, H, W)
            else:
                # color_tensor: (H, W, 3)
                H, W, _ = rgb_array.shape
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
            labels, edges, _, captions = make_vlm_edges_and_captions(
                rgb_array,
                curr_det,
                self.obj_classes,
                detection_class_labels,
                self.det_exp_vis_path,
                color_path,
                self.cfg.make_edges,
                self.openai_client,
                self.frame_idx,
                save_result=True)
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
            rgb_array_ = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
            image_crops, image_feats, text_feats = compute_clip_features_batched(
                rgb_array_, curr_det, self.clip_model,
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
                "labels": labels,
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
                    rgb_array, curr_det, self.obj_classes.get_classes_arr())
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
        self.prev_adjusted_pose = orr_log_camera(self.intrinsics, camera_pose,
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
        # resized_grounded_obs = resize_gobs(raw_grounded_obs, rgb_array)
        resized_grounded_obs = raw_grounded_obs
        ##### 1.5. [시작]  프레임 내 segmentation 결과 필터링
        """
 **필터링 기준 설정**:
    매우 작거나(25) + 배경을 제거하거나 (skip_bg=True) + 이미지 크기의 50% 이상 물체 제거
        """
        filtered_grounded_obs = filter_gobs(
            resized_grounded_obs,
            rgb_array,
            skip_bg=self.cfg.skip_bg,
            # ["wall", "floor", "ceiling"]
            BG_CLASSES=self.obj_classes.get_bg_classes_arr(),
            mask_area_threshold=self.cfg.mask_area_threshold,  # 25
            max_bbox_area_ratio=self.cfg.max_bbox_area_ratio,  # 0.5
            mask_conf_threshold=None,  #self.cfg.mask_conf_threshold, # 0.25
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
        #### 2. pcd 처리

        ##### 2.1. [시작] 3d pointcloud 만들기
        # obj_pcds_and_bboxes : [ {'pcd': pcd, 'bbox': bbox} , ... ]
        obj_pcds_and_bboxes: List[Dict[str, Any]] = measure_time(
            detections_to_obj_pcd_and_bbox)(
                depth_array=depth_array,
                masks=grounded_obs['mask'],
                cam_K=self.intrinsics[:3, :3],  # Camera intrinsics
                image_rgb=rgb_array_,
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

        #### 3. 기존 object pcd와 융합
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

        ##### 4.1. [시작] 주기적 "누적 object" 후처리

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
                device=self.args.device,
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

    def _get_pose_data(self, time_msg: Time) -> Optional[np.ndarray]:
        try:
            vl_transform = self._tf_buffer.lookup_transform(
                target_frame=self._target_frame,
                source_frame=self._source_frame,
                time=time_msg,
                timeout=rclpy.duration.Duration(seconds=0.3))
            agent_pose = self._transform_stamped_to_matrix(vl_transform)

            return agent_pose
        except LookupException:
            print("[pose tf listener]LookupException")
        except ConnectivityException:
            print("[pose tf listener]ConnectivityException")
        except ExtrapolationException:
            print("[pose tf listener]ExtrapolationException")
        return None

    def rgb_callback(self, msg: CompressedImage) -> Tuple[np.ndarray, Time]:
        builtin_time = msg.header.stamp
        # 메시지에서 이미지 데이터를 읽어서 OpenCV 이미지로 변환
        np_array = np.frombuffer(msg.data, np.uint8)
        rgb_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # # publisher가 BGR 순서의 이미지를 보내고 있음.
        # # frombuffer를 사용하면,
        # # 바이너리 데이터를 복사하지 않고, 직접 배열로 변환할 수 있어 메모리 사용량을 최소화
        # rgb_array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_array, builtin_time

    def color_camera_info_callback(
            self, msg: CameraInfo) -> Tuple[np.ndarray, np.ndarray]:
        # get distortion coefficients and camera matrix.
        dist_coeffs = np.array(msg.d)  # (5,)
        camera_matrix = np.array(msg.k).reshape(3, 3)
        """
self.rgb_intrinsics: [[     301.56           0      214.37]
 [          0       300.9      121.26]
 [          0           0           1]]
self.rgb_dist_coeffs: [          0           0           0           0           0]

        """
        self.rgb_intrinsics = camera_matrix
        self.rgb_dist_coeffs = dist_coeffs
        return camera_matrix, dist_coeffs

    def depth_camera_info_callback(
            self, msg: CameraInfo) -> Tuple[np.ndarray, np.ndarray]:
        # get distortion coefficients and camera matrix.
        dist_coeffs = np.array(msg.d)  # (5,) # TODO: 전부 0으로 나오므로, 의미가 없음.
        camera_matrix = np.array(msg.k).reshape(3, 3)
        """
self.intrinsics: [[     209.05           0      212.36]
 [          0      209.05      118.82]
 [          0           0           1]]        
        """
        self.intrinsics = camera_matrix
        """
self.depth_dist_coeffs: [          0           0           0           0           0]
        """
        self.depth_dist_coeffs = dist_coeffs
        return camera_matrix, dist_coeffs

    def depth_callback(self,
                       msg: CompressedImage,
                       rescale_depth: float = 4.) -> Tuple[np.ndarray, Time]:
        # np_array = np.frombuffer(msg.data, np.uint8)
        # depth_array = cv2.imdecode(np_array, cv2.IMREAD_ANYDEPTH)
        # TODO: check
        builtin_time = msg.header.stamp
        img = np.ndarray(shape=(1, len(msg.data)),
                         dtype="uint8",
                         buffer=msg.data)
        img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR) * rescale_depth / 255.0
        return img, builtin_time


@hydra.main(version_base=None,
            config_path="../hydra_configs/",
            config_name="rerun_realtime_mapping_ros")
def main(cfg: DictConfig):

    parser = argparse.ArgumentParser(
        description="RealtimeHumanSegmenterNode Node")
    parser.add_argument('--realsense_idx',
                        type=int,
                        default=0,
                        help="Robot ID for the RealtimeHumanSegmenterNode Node")
    parser.add_argument('--use_world_model',
                        type=bool,
                        default=False,
                        help="Use YOLO-World model for detection")
    parser.add_argument('--debug_mode',
                        type=bool,
                        default=False,
                        help="Save the results to debug")
    parser.add_argument('--min_points_threshold',
                        type=int,
                        default=16,
                        help="Minimum number of points required for an object")
    parser.add_argument('--conf',
                        type=float,
                        default=0.2,
                        help="Confidence threshold for detection")
    parser.add_argument(
        '--obj_pcd_max_points',
        type=int,
        default=5000,
        help="Maximum number of points in an object's point cloud")
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help="Device to use for computation (cpu or cuda)")
    args = parser.parse_args()

    # ROS2 초기화
    rclpy.init()
    ros2_bridge = None
    try:
        ros2_bridge = RealtimeHumanSegmenterNode(cfg, args)
        rclpy.spin(ros2_bridge)
    except:
        traceback.print_exc()
        if ros2_bridge is not None:
            ros2_bridge.wrap_up()
            ros2_bridge.vis.destroy_window()
            ros2_bridge.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":

    main()
