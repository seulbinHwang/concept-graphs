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
"""
이 함수는 **Hydra**라는 도구를 사용해 **설정 파일을 기반으로 작업을 실행**하는 역할
    - `@hydra.main`이라는 데코레이터를 사용해 Hydra 설정 파일을 불러와 프로그램의 실행 환경을 설정

### 1. **주요 역할**
- `config_path`와 `config_name`에 지정된 설정 파일을 불러와서, 그 설정에 따라 프로그램을 실행 

### 2. **세부 로직**
- **Hydra 데코레이터 사용**: `@hydra.main`은 Hydra 설정을 적용하는 데 사용됩니다. 
    - 여기서 설정 파일은 `../hydra_configs/` 폴더 안에 있으며, 
        - `rerun_realtime_mapping.yaml` 파일이 사용됩니다.
- **`cfg` 객체**: 
    - 이 설정 파일의 내용이 `cfg`라는 **구성 객체**(여기서는 `DictConfig` 타입)로 전달
        - 이 객체를 통해 설정값을 접근하고, 프로그램의 동작을 제어할 수 있습니다.

"""


# A logger for this file
@hydra.main(version_base=None,
            config_path="../hydra_configs/",
            config_name="rerun_realtime_mapping")
# @profile
def main(cfg: DictConfig):
    tracker = MappingTracker()

    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    orr.spawn()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(
        project="concept-graphs",
        #    entity="concept-graphs",
        config=cfg_to_dict(cfg),
    )
    cfg = process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        # dataset/dataconfigs/replica/replica.yaml
        dataconfig=cfg.dataset_config,
        # Replica
        basedir=cfg.dataset_root,
        # room0
        sequence=cfg.scene_id,
        start=cfg.start,  # 0
        end=cfg.end,  # -1
        stride=cfg.stride,  # 50
        desired_height=cfg.image_height,  # None # 680
        desired_width=cfg.image_width,  # None # 1200
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # # For visualization
    # if cfg.vis_render:
    #     # render a frame, if needed (not really used anymore since rerun)
    #     view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
    #     obj_renderer = OnlineObjectRenderer(
    #         view_param=view_param,
    #         base_objects=None,
    #         gray_map=False,
    #     )
    #     frames = []
    # output folder for this mapping experiment
    # dataset_root: Datasets
    # scene_id: Replica/room0
    # exp_suffix: r_mapping_stride10
    # exp_out_path: Datasets/Replica/room0/exps/r_mapping_stride10
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id,
                                    cfg.exp_suffix)

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root,
                                    cfg.scene_id,
                                    cfg.detections_exp_suffix,
                                    make_dir=False)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'],
        bg_classes=detections_exp_cfg[
            'bg_classes'],  # "wall", "floor", "ceiling"
        skip_bg=detections_exp_cfg['skip_bg'])  # False

    # if we need to do detections
    # det_exp_path:
    # concept-graphs/Datasets/Replica/room0/exps/s_detections_stride10
    print("det_exp_path:", det_exp_path)
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    print("run_detections:", run_detections)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)

    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)('yolov8l-world.pt')
        sam_predictor = SAM(
            'sam_b.pt')  # SAM('mobile_sam.pt') # UltraLytics SAM
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())
        if RUN_OPEN_API:
            openai_client = get_openai_client()
        else:
            openai_client = None

    else:
        print("\n".join(["NOT Running detections..."] * 10))

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg,
                      exp_out_path,
                      is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0

    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter += 1
        orr.set_time_sequence("frame", frame_idx)

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)  # 필요 없음
        # color and depth tensors, and camera instrinsics matrix
        """
        color_path -> image_original_pil (PIL) # 필요 없음
        color_path -> image_rgb (cv2)
            - run_detections가 True일 때, 아래 image_rgb 를 대체
        color_tensor: (680, 1200, 3) -> image_rgb: (680, 1200, 3) # resized
        depth_tensor: (680, 1200, 1) -> depth_array: (680, 1200) # resized
        intrinsics: (4, 4)  # resize 가 들어간 것
        """
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]  # (H, W)
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy()  # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        vis_save_path_for_vlm = get_vlm_annotated_image_path(
            det_exp_vis_path, color_path)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(
            det_exp_vis_path, color_path, w_edges=True)

        if run_detections:
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path))  # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (H, W, 3)

            # Do initial object detection (YOLO world)

            ##### 1. [시작] RGBD에서 instance segmentation 진행
            results = detection_model.predict(color_path,
                                              conf=0.1,
                                              verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(
                int)  # (N,)
            # detection_class_labels = [ "sofa chair 0", ... ]
            detection_class_labels = [
                f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
                for class_idx, class_id in enumerate(detection_class_ids)
            ]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                # segmentation
                sam_out = sam_predictor.predict(color_path,
                                                bboxes=xyxy_tensor,
                                                verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()  # (N, H, W)
            else:
                masks_np = np.empty((0, *color_tensor.shape[:2]),
                                    dtype=np.float64)
            # Create a detections object that we will save later.
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            ##### 1. [끝] RGBD에서 instance segmentation 진행

            # Make the edges
            # detection_class_labels: ["sofa chair 0", ...]
            # det_exp_path:
            # Datasets/Replica/room0/exps/s_detections_stride10
            # det_exp_vis_path:
            # Datasets/Replica/room0/exps/s_detections_stride10/vis
            # color_path
            # Datasets/Replica/room0/results/frame000000.jpg
            # cfg.make_edges: True
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

            ##### 5. [시작] VLM을 통해 edge와 caption 정보를 구한다.
            ##### 논문과 다르게, 현 이미지 1장에 대한 edge와 node caption 정보를 구한다.
            labels, edges, edge_image, captions = make_vlm_edges_and_captions(
                image, curr_det, obj_classes, detection_class_labels,
                det_exp_vis_path, color_path, cfg.make_edges, openai_client)
            del edge_image
            ##### 5. [끝] VLM을 통해 edge와 caption 정보를 구한다.
            """
        image_crops: List[Image.Image]
            - 잘라낸 이미지들의 리스트
        image_feats: np.ndarray
            - 잘라낸 이미지들의 CLIP feature: shape (N, 512)
        text_feats: List
            - 빈 리스트
            """
            ##### 2. [시작] CLIP feature를 계산
            # image_rgb: (H, W, 3) 원본 사이즈
            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess,
                clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)
            ##### 2. [끝] CLIP feature를 계산

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection
                "xyxy": curr_det.xyxy,  # (34, 4)
                "confidence": curr_det.confidence,  # (34,)
                "class_id": curr_det.class_id,  # (34,)
                "mask": curr_det.mask,  # (34, 680, 1200)
                "classes":
                    obj_classes.get_classes_arr(),  # len = 200, "alarm clock"
                "image_crops": image_crops,  # len = 34, <PIL.Image.Image>
                "image_feats": image_feats,  # (34, 1024)
                "text_feats": text_feats,  # len = 0 # 아마?
                "detection_class_labels":
                    detection_class_labels,  # len = 34, "sofa chair 0"
                "labels": labels,  # len = 19, "sofa chair 0"
                "edges": edges,  # len = 0
                "captions": captions,  # len = 0
            }
            raw_grounded_obs = results

            # save the detections if needed
            # important
            if cfg.save_detections:
                # det_exp_vis_path:
                # Datasets/Replica/room0/exps/s_detections_stride10/vis
                # color_path
                # Datasets/Replica/room0/results/frame000000.jpg
                # vis_save_path:
                # Datasets/Replica/room0/exps/s_detections_stride10/vis/frame000000.jpg
                """ 4장 그림 그려서 저장하는 과정임
                vis_save_path: bounding box와 mask가 모두 그려진 이미지
                
                """
                vis_save_path = (det_exp_vis_path /
                                 color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(
                    image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)
                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255,
                                                cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb,
                                               cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(
                    depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(
                    str(vis_save_path).replace(".jpg", "_depth.jpg"),
                    annotated_depth_image)
                cv2.imwrite(
                    str(vis_save_path).replace(".jpg", "_depth_only.jpg"),
                    depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem,
                                       results)
        else:
            # Support current and old saving formats
            print("color_path:", color_path)
            print("color_path.stem:", color_path.stem)
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_grounded_obs = load_saved_detections(det_exp_pkl_path /
                                                         color_path.stem)
            elif os.path.exists(det_exp_pkl_path /
                                f"{int(color_path.stem):06}"):
                raw_grounded_obs = load_saved_detections(
                    det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                # if no detections, throw an error
                raise FileNotFoundError(
                    f"No detections found for frame {frame_idx}at paths \n{det_exp_pkl_path / color_path.stem} or \n{det_exp_pkl_path / f'{int(color_path.stem):06}'}."
                )

        ########

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()  # (4, 4)

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose  # (4, 4)
        # orr = Optional re-run
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose,
                                            prev_adjusted_pose, cfg.image_width,
                                            cfg.image_height, frame_idx)

        orr_log_rgb_image(color_path)
        orr_log_annotated_image(color_path, det_exp_vis_path)
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
                    obj_classes.get_classes_arr(),  # len = 200, "alarm clock"
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
        resized_grounded_obs = resize_gobs(raw_grounded_obs, image_rgb)
        ##### 1.1. [시작] segmentation 결과 필터링
        """
2. **필터링 기준 설정**:
     - **마스크 면적**: 
        - 마스크의 면적이 `mask_area_threshold`보다 작은 객체는 필터링 ( 25 )
     - **배경 클래스 필터링**: 
        - `skip_bg`가 활성화된 경우, 배경 클래스(`BG_CLASSES`)에 속하는 객체는 필터링
     - **경계 상자 면적 비율**: 
        - 경계 상자 면적이 이미지의 `max_bbox_area_ratio(0.5)`를 초과하는 객체는 필터링
     - **신뢰도 필터링**: 
        - 객체의 신뢰도가 `mask_conf_threshold`보다 낮은 경우 해당 객체는 필터링
        """
        filtered_grounded_obs = filter_gobs(
            resized_grounded_obs,
            image_rgb,
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )
        ##### 1.1. [끝] segmentation 결과 필터링

        grounded_obs = filtered_grounded_obs

        if len(grounded_obs['mask']) == 0:  # no detections in this frame
            continue

        # this helps make sure things like 베개 on 소파 are separate objects.
        grounded_obs['mask'] = mask_subtract_contained(grounded_obs['xyxy'],
                                                       grounded_obs['mask'])
        """
-----------all shapes of detections_to_obj_pcd_and_bbox inputs:--------
depth_array.shape: (680, 1200)
grounded_obs['mask'].shape: (N, 680, 1200)
intrinsics.cpu().numpy()[:3, :3].shape: (3, 3)
image_rgb.shape: (680, 1200, 3)
adjusted_pose.shape: (4, 4)
        """
        ##### 3. [시작] pointclouds 만들기
        # obj_pcds_and_bboxes : [ {'pcd': pcd, 'bbox': bbox} , ... ]
        obj_pcds_and_bboxes: List[Dict[str, Any]] = measure_time(
            detections_to_obj_pcd_and_bbox)(
                depth_array=depth_array,
                masks=grounded_obs['mask'],
                cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
                image_rgb=image_rgb,  # None 으로 해도 됨 (pointcloud 색깔 구하려는 것)
                trans_pose=adjusted_pose,
                min_points_threshold=cfg.min_points_threshold,  # 16
                # overlap # "iou", "giou", "overlap"
                spatial_sim_type=cfg.spatial_sim_type,  # overlap
                obj_pcd_max_points=cfg.obj_pcd_max_points,  # 5000
                device=cfg.device,
            )

        for obj in obj_pcds_and_bboxes:
            if obj:
                # obj: {'pcd': pcd, 'bbox': bbox}
                """
                포인트 클라우드를 voxel (0.01m)로 다운샘플링하고,
                dbscan clustering기반으로
                    노이즈를 제거하여 클러스터링을 통해 중요한 포인트만 남기는 것
                    0.1m 거리 내의 cluster을 모으는데, 최소 10개 포인트가 있어야 한다.
                """
                ##### 3.1. [시작] pointclouds 필터링
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],  # 0.01
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],  # True
                    dbscan_eps=cfg["dbscan_eps"],  # 0.1
                    dbscan_min_points=cfg["dbscan_min_points"],  # 10
                )
                # point cloud를 filtering 했으니, bounding box를 다시 계산
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'],  # overlap
                    pcd=obj["pcd"],
                )
                ##### 3.1. [끝] pointclouds 필터링
        """
        obj_pcds_and_bboxes : [ {'pcd': pcd, 'bbox': bbox} , ... ]
            grounded_obs = {
                # add new uuid for each detection
                "xyxy": curr_det.xyxy,  # (34, 4)
                "confidence": curr_det.confidence,  # (34,)
                "class_id": curr_det.class_id,  # (34,)
                "mask": curr_det.mask,  # (34, 680, 1200)
                "classes":
                    obj_classes.get_classes_arr(),  # len = 200, "alarm clock"
                "image_crops": image_crops,  # len = 34, <PIL.Image.Image>
                "image_feats": image_feats,  # (34, 1024)
                "text_feats": text_feats,  # len = 0 # 아마?
                "detection_class_labels":
                    detection_class_labels,  # len = 34, "sofa chair 0"
                "labels": labels,  # len = 19, "sofa chair 0"
                "edges": edges,  # len = 0
                "captions": captions,  # len = 0
            }
        color_path: Datasets/Replica/room0/results/frame000000.jpg
        
        """
        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, grounded_obs, color_path, obj_classes,
            frame_idx)

        if len(detection_list) == 0:  # no detections, skip
            continue
        ##### 3. [끝] pointclouds 만들기

        ##### 4. [시작] 기존 object 들과 융합하기

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log({
                "total_objects_so_far": tracker.get_total_objects(),
                "objects_this_frame": len(detection_list),
            })
            continue

        ### compute similarities and then merge
        # spatial_sim : (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 중첩 비율
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'],  # overlap
            detection_list=detection_list,
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size'])  # 0.01
        # visual_sim :  (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 코싸인 유사도
        visual_sim = compute_visual_similarities(detection_list, objects)

        # match_method = "sim_sum" # "sep_thresh", "sim_sum"
        # phys_bias = 0.0
        # 단순하게 두개를 그냥 더함
        # agg_sim : (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 유사도
        agg_sim = aggregate_similarities(match_method=cfg['match_method'],
                                         phys_bias=cfg['phys_bias'],
                                         spatial_sim=spatial_sim,
                                         visual_sim=visual_sim)

        # Perform matching of detections to existing objects
        # match_indices: 길이는 "새 검지 개수"
        match_indices: List[Optional[int]] = match_detections_to_objects(
            agg_sim=agg_sim,
            detection_threshold=cfg['sim_threshold']  # 1.2
        )

        objects = merge_obj_matches(
            detection_list=detection_list,
            objects=objects,
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'],  # 0.01
            dbscan_remove_noise=cfg['dbscan_remove_noise'],  # True
            dbscan_eps=cfg['dbscan_eps'],  # 0.1
            dbscan_min_points=cfg['dbscan_min_points'],  # 10
            spatial_sim_type=cfg['spatial_sim_type'],  # overlap
            device=cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )
        ##### 4. [끝] 기존 object 들과 융합하기

        # fix the class names for objects
        # they should be the most popular name, not the first name
        for idx, obj in enumerate(objects):
            temp_class_name = obj["class_name"]  # "sofa chair"
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(
                1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr(
            )[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        ##### 5. [시작] edge 계산하기

        map_edges = process_edges(match_indices, grounded_obs, len(objects),
                                  objects, map_edges, frame_idx)
        is_final_frame = frame_idx == len(dataset) - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # Clean up outlier edges
        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            obj1_class_name = objects[curr_obj1_idx]['class_name']
            obj2_class_name = objects[curr_obj2_idx]['class_name']
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections
            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])
        ### Perform post-processing periodically if told so

        ##### 6. [시작] 주기적 "누적 object" 후처리

        # Denoising
        if processing_needed(
                # Run DBSCAN every k frame. This operation is heavy
                cfg["denoise_interval"],  # 20
                cfg["run_denoise_final_frame"],  # True
                frame_idx,
                is_final_frame,
        ):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'],  # 0.01
                dbscan_remove_noise=cfg['dbscan_remove_noise'],  # True
                dbscan_eps=cfg['dbscan_eps'],  # 0.1
                dbscan_min_points=cfg['dbscan_min_points'],  # 10
                spatial_sim_type=cfg['spatial_sim_type'],  # overlap
                device=cfg['device'],
                objects=objects)

        # Filtering
        if processing_needed(
                # Filter objects that have too few associations or are too small
                cfg["filter_interval"],  # 5
                cfg["run_filter_final_frame"],  # True
                frame_idx,
                is_final_frame,
        ):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'],
                obj_min_detections=cfg['obj_min_detections'],
                objects=objects,
                map_edges=map_edges)

        # Merging
        if processing_needed(
                # Merge objects based on geometric and semantic similarity
                cfg["merge_interval"],  # 5
                cfg["run_merge_final_frame"],  # True
                frame_idx,
                is_final_frame,
        ):
            objects, map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=cfg["make_edges"],
                map_edges=map_edges)
        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        orr_log_edges(objects, map_edges, obj_classes)

        ##### 6. [끝] 주기적 "누적 object" 후처리

        if cfg.save_objects_all_frames:
            save_objects_for_frame(obj_all_frames_out_path, frame_idx, objects,
                                   cfg.obj_min_detections, adjusted_pose,
                                   color_path)

        if cfg.vis_render:  # False
            # render a frame, if needed (not really used anymore since rerun)
            vis_render_image(
                objects,
                obj_classes,
                obj_renderer,
                image_original_pil,
                adjusted_pose,
                frames,
                frame_idx,
                color_path,
                cfg.obj_min_detections,
                cfg.class_agnostic,
                cfg.debug_render,
                is_final_frame,
                cfg.exp_out_path,
                cfg.exp_suffix,
            )
        # periodically_save_pcd: False
        if cfg.periodically_save_pcd and (
                counter % cfg.periodically_save_pcd_interval == 0):
            # save the pointcloud
            save_pointcloud(exp_suffix=cfg.exp_suffix,
                            exp_out_path=exp_out_path,
                            cfg=cfg,
                            objects=objects,
                            obj_classes=obj_classes,
                            latest_pcd_filepath=cfg.latest_pcd_filepath,
                            create_symlink=True)

        owandb.log({
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
            "total_objects": tracker.get_total_objects(),
            "objects_this_frame": len(objects),
            "total_detections": tracker.get_total_detections(),
            "detections_this_frame": len(detection_list),
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })
    # LOOP OVER -----------------------------------------------------

    # Consolidate captions

    ##### 7. [시작] 각 물체마다, caption 합치기
    for object in objects:
        obj_captions = object['captions'][:20]  # 첫 20개만 사용
        consolidated_caption = consolidate_captions(openai_client, obj_captions)
        object['consolidated_caption'] = consolidated_caption
    ##### 7. [끝] 각 물체마다, caption 합치기

    # exp_suffix: r_mapping_stride10
    # exp_out_path: exps/r_mapping_stride10/
    handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix,
                        exp_out_path)

    # Save the pointcloud
    if cfg.save_pcd:  # True
        # exp_suffix: rerun_r_mapping_stride10
        # exp_out_path: exps/r_mapping_stride10/
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            # ./latest_pcd_save
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges)

    if cfg.save_json:
        save_obj_json(exp_suffix=cfg.exp_suffix,
                      exp_out_path=exp_out_path,
                      objects=objects)

        save_edge_json(exp_suffix=cfg.exp_suffix,
                       exp_out_path=exp_out_path,
                       objects=objects,
                       edges=map_edges)

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump(
                {
                    'cfg': cfg,
                    'class_names': obj_classes.get_classes_arr(),
                    'class_colors': obj_classes.get_class_color_dict_by_index(),
                }, f)

    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    owandb.finish()


if __name__ == "__main__":
    main()
