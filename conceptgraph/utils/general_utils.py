from copy import deepcopy
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Tuple, List, Dict
import pickle
# from conceptgraph.utils.vis import annotate_for_vlm, filter_detections, plot_edges_from_vlm
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import prepare_objects_save_vis
from conceptgraph.utils.ious import mask_subtract_contained
import supervision as sv
import scipy.ndimage as ndi
from conceptgraph.utils.vlm import get_obj_captions_from_image_gpt4v, get_obj_rel_from_image_gpt4v, vlm_extract_object_captions
import cv2
import re

from omegaconf import OmegaConf
import torch
import numpy as np
import time


class Timer:

    def __init__(self, heading="", verbose=True):
        self.verbose = verbose
        if not self.verbose:
            return
        self.heading = heading

    def __enter__(self):
        if not self.verbose:
            return self
        self.start = time.time()
        return self

    def __exit__(self, *args):
        if not self.verbose:
            return
        self.end = time.time()
        self.interval = self.end - self.start
        print(self.heading, self.interval)


def prjson(input_json, indent=0):
    """ Pretty print a json object """
    if not isinstance(input_json, list):
        input_json = [input_json]

    print("[")
    for i, entry in enumerate(input_json):
        print("  {")
        for j, (key, value) in enumerate(entry.items()):
            terminator = "," if j < len(entry) - 1 else ""
            if isinstance(value, str):
                formatted_value = value.replace("\\n",
                                                "\n").replace("\\t", "\t")
                print('    "{}": "{}"{}'.format(key, formatted_value,
                                                terminator))
            else:
                print(f'    "{key}": {value}{terminator}')
        print("  }" + ("," if i < len(input_json) - 1 else ""))
    print("]")


def cfg_to_dict(input_cfg):
    """ Convert a Hydra configuration object to a native Python dictionary,
    ensuring all special types (e.g., ListConfig, DictConfig, PosixPath) are
    converted to serializable types for JSON. Checks for non-serializable objects. """

    def convert_to_serializable(obj):
        """ Recursively convert non-serializable objects to serializable types. """
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def check_serializability(obj, context=""):
        """ Attempt to serialize the object, raising an error if not possible. """
        try:
            json.dumps(obj)
        except TypeError as e:
            raise TypeError(
                f"Non-serializable object encountered in {context}: {e}")

        if isinstance(obj, dict):
            for k, v in obj.items():
                check_serializability(
                    v, context=f"{context}.{k}" if context else str(k))
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                check_serializability(item, context=f"{context}[{idx}]")

    # Convert Hydra configs to native Python types
    # check if its already a dictionary, in which case we don't need to convert it
    if not isinstance(input_cfg, dict):
        native_cfg = OmegaConf.to_container(input_cfg, resolve=True)
    else:
        native_cfg = input_cfg
    # Convert all elements to serializable types
    serializable_cfg = convert_to_serializable(native_cfg)
    # Check for serializability of the entire config
    check_serializability(serializable_cfg)

    return serializable_cfg


def get_stream_data_out_path(dataset_root, scene_id, make_dir=True):
    stream_data_out_path = Path(dataset_root) / scene_id
    stream_rgb_path = stream_data_out_path / "rgb"
    stream_depth_path = stream_data_out_path / "depth"
    stream_poses_path = stream_data_out_path / "poses"

    if make_dir:
        stream_rgb_path.mkdir(parents=True, exist_ok=True)
        stream_depth_path.mkdir(parents=True, exist_ok=True)
        stream_poses_path.mkdir(parents=True, exist_ok=True)

    return stream_rgb_path, stream_depth_path, stream_poses_path


def get_exp_out_path(dataset_root, scene_id, exp_suffix, make_dir=True):
    exp_out_path = Path(dataset_root) / scene_id / "exps" / f"{exp_suffix}"
    if make_dir:
        exp_out_path.mkdir(exist_ok=True, parents=True)
    return exp_out_path


def get_vis_out_path(exp_out_path):
    vis_folder_path = exp_out_path / "vis"
    vis_folder_path.mkdir(exist_ok=True, parents=True)
    return vis_folder_path


def get_det_out_path(exp_out_path, make_dir=True):
    detections_folder_path = exp_out_path / "detections"
    if make_dir:
        detections_folder_path.mkdir(exist_ok=True, parents=True)
    return detections_folder_path


def check_run_detections(force_detection, det_exp_path):
    # first check if det_exp_path directory exists
    if force_detection:
        return True
    if not det_exp_path.exists():
        return True
    return False


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def annotate_for_vlm(
        image: np.ndarray,
        detections: sv.Detections,
        obj_classes,
        labels: List[str],
        save_path=None,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        text_color: tuple = (255, 255, 255),
        text_scale: float = 0.6,
        text_thickness: int = 2,
        text_bg_color: tuple = (255, 255, 255),
        text_bg_opacity:
    float = 0.95,  # Opacity from 0 (transparent) to 1 (opaque)
        small_mask_threshold=0.002,
        mask_opacity: float = 0.2  # Opacity for mask fill
) -> np.ndarray:
    annotated_image = image.copy()

    # if image.shape[0] > 700:
    #     print(f"Line 604, image.shape[0]: {image.shape[0]}")
    #     text_scale = 2.5
    #     text_thickness = 5
    total_pixels = image.shape[0] * image.shape[1]
    small_mask_size = total_pixels * small_mask_threshold

    detections_mask = detections.mask
    detections_mask = mask_subtract_contained(detections.xyxy, detections_mask)

    # Sort detections by mask area, large to small, and keep track of original indices
    mask_areas = [np.count_nonzero(mask) for mask in detections_mask]
    sorted_indices = sorted(range(len(mask_areas)),
                            key=lambda x: mask_areas[x],
                            reverse=True)

    # Iterate over each mask and corresponding label in the detections in sorted order
    for i in sorted_indices:
        mask = detections_mask[i]
        label = labels[i]
        label_num = label.split(" ")[-1]
        label_name = re.sub(r'\s*\d+$', '', label).strip()
        bbox = detections.xyxy[i]

        obj_color = obj_classes.get_class_color(int(detections.class_id[i]))
        # multiply by 255 to convert to BGR
        obj_color = tuple([int(c * 255) for c in obj_color])

        # Add color over mask for this object
        mask_uint8 = mask.astype(np.uint8)
        mask_color_image = np.zeros_like(annotated_image)
        mask_color_image[mask_uint8 > 0] = obj_color
        # cv2.addWeighted(annotated_image, 1, mask_color_image, mask_opacity, 0, annotated_image)

        # Draw contours
        contours, _ = cv2.findContours(mask_uint8 * 255, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_image, contours, -1, obj_color, thickness)

        # Determine if the mask is considered "small"
        if mask_areas[i] < small_mask_size:
            x_center = int(
                bbox[2])  # Place the text to the right of the bounding box
            y_center = int(
                bbox[1])  # Place the text above the top of the bounding box
        else:
            # Calculate the centroid of the mask
            ys, xs = np.nonzero(mask)
            y_center, x_center = ndi.center_of_mass(mask)
            x_center, y_center = int(x_center), int(y_center)

        # Prepare text background
        text = label_num + ": " + label_name
        (text_width,
         text_height), baseline = cv2.getTextSize(text,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  text_scale, text_thickness)
        text_x_left = x_center - text_width // 2
        text_y_top = y_center + (text_height) // 2

        # Create a rectangle sub-image for the text background
        b_pad = 2  # background rectangle padding
        rect_top_left = (text_x_left - b_pad,
                         text_y_top - text_height - baseline - b_pad)
        rect_bottom_right = (text_x_left + text_width + b_pad,
                             text_y_top - baseline // 2 + b_pad)
        sub_img = annotated_image[rect_top_left[1]:rect_bottom_right[1],
                                  rect_top_left[0]:rect_bottom_right[0]]

        # Create the background rectangle with the specified color and opacity
        # make the text bg color be the negative of the text color
        text_bg_color = tuple([255 - c for c in obj_color])
        # now make text bg color grayscale
        text_bg_color = tuple([int(sum(text_bg_color) / 3)] * 3)
        background_rect = np.full(sub_img.shape, text_bg_color, dtype=np.uint8)
        # cv2.addWeighted(sub_img, 1 - text_bg_opacity, background_rect, text_bg_opacity, 0, sub_img)

        # Draw text with background
        cv2.putText(
            annotated_image,
            text,
            (text_x_left, text_y_top - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            # obj_color,
            # (255,255,255),
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA)

        # Draw text with background
        cv2.putText(
            annotated_image,
            text,
            (text_x_left, text_y_top - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            # (0,0,0),
            obj_color,
            text_thickness - 1,
            cv2.LINE_AA)

        if save_path:
            cv2.imwrite(save_path, annotated_image)

    return annotated_image, sorted_indices


def plot_edges_from_vlm(image: np.ndarray,
                        edges,
                        detections: sv.Detections,
                        obj_classes,
                        labels: List[str],
                        sorted_indices: List[int],
                        save_path=None) -> np.ndarray:
    annotated_image = image.copy()

    # Create a map from label to mask centroid and color for quick lookup
    label_to_centroid_color = {}
    for idx in sorted_indices:
        mask = detections.mask[idx]
        label_num = labels[idx].split(' ')[
            -1]  # Assuming label format is 'object X'
        obj_color = obj_classes.get_class_color(int(detections.class_id[idx]))
        obj_color = tuple([int(c * 255) for c in obj_color])  # Convert to BGR

        # Determine the centroid of the mask
        ys, xs = np.nonzero(mask)
        if ys.size > 0 and xs.size > 0:
            y_center, x_center = ndi.center_of_mass(mask)
            centroid = (int(x_center), int(y_center))
        else:
            # Fallback to bbox center if mask is empty
            bbox = detections.xyxy[idx]
            centroid = (int(
                (bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

        label_to_centroid_color[label_num] = (centroid, obj_color)

    # Draw edges based on relationships specified
    for edge in edges:
        src_label, _, dst_label = edge
        src_label = str(src_label)  # Assuming label is int of object_index
        dst_label = str(dst_label)
        if src_label in label_to_centroid_color and dst_label in label_to_centroid_color:
            src_centroid, _ = label_to_centroid_color[src_label]
            dst_centroid, dst_color = label_to_centroid_color[dst_label]
            # Draw line from source to destination object with the color of the destination object
            cv2.line(annotated_image, src_centroid, dst_centroid, dst_color, 2)

    if save_path:
        cv2.imwrite(str(save_path), annotated_image)

    return annotated_image


def filter_detections(
    image,
    detections: sv.Detections,
    classes,
    top_x_detections=None,
    confidence_threshold: float = 0.0,
    given_labels=None,
    iou_threshold: float = 0.80,  # IoU similarity threshold
    proximity_threshold: float = 20.0,  # Default proximity threshold
    keep_larger:
    bool = True,  # Keep the larger bounding box by area if True, else keep the smaller
    min_mask_size_ratio=0.00025
) -> Tuple[sv.Detections, List[str]]:
    """
    
    Args:
        image: (H, W, 3)
        detections: sv.Detections
        classes: ObjectClasses
        top_x_detections: 150000
        confidence_threshold: 0.00001
        given_labels: [ "sofa chair 0", ... ]
        iou_threshold: 0.80
        proximity_threshold: 20.0 
        keep_larger: True
        min_mask_size_ratio: 0.00025 

    Returns:
        Tuple[sv.Detections, List[str]]: filtered_detections, filtered_labels
이 함수는 객체 탐지 후 필터링

### 입력 데이터 처리:
2. **탐지 결과 정렬**: 
    `detections` 객체의  `confidence` 값을 기준으로 내림차순으로 정렬
     `top_x_detections` 값이 주어지면, 상위 `X`개의 탐지 결과만 남깁니다.

### 필터링 로직:
- **탐지 결과 간 근접성 및 겹침 필터링**:
  1. **마스크 크기 평가**: 
    (탐지된 객체의 마스크 크기를 평가하여,)
        (`min_mask_size_ratio`로 설정된 임계값보다 작은 마스크를 가진 객체는 제거)

  2. **IoU(Intersection over Union)**: 
    각 객체의 마스크 간 겹침 정도(IoU)를 계산하여, 
        `iou_threshold` 이상의 겹침을 가진 경우 탐지 결과를 필터링
    탐지 객체의 클래스 이름과 겹치는 객체의 클래스 이름을 출력

  3. **중심점 거리 계산**: 
    객체 간의 바운딩 박스 중심점 거리(proximity)를 계산하여, 
    `proximity_threshold` 이내로 가까운 객체는 제거
    근접성이 판단 기준일 때는 객체의 크기(면적)에 따라, 
    `keep_larger` 플래그에 따라 더 큰 또는 더 작은 객체를 남기고 나머지는 제거
  
- **배경 클래스 필터링**: 
    탐지된 객체가 배경 클래스(`bg_classes`)에 속하는 경우 이를 필터링하여 제거

### 결과 생성:
- 필터링된 탐지 결과에서 신뢰도, 클래스 ID, 좌표(xyxy), 마스크 정보를 추출하여 `sv.Detections` 객체를 다시 생성
- 필터링된 탐지 객체들의 레이블 리스트도 함께 반환

    """
    """
- 남은 로직
  - IoU가 80% 이상 겹치면, 신뢰도가 낮은 객체를 제거
  - bg_classes 클래스 제거
    """
    if not (hasattr(detections, 'confidence') and
            hasattr(detections, 'class_id') and hasattr(detections, 'xyxy')):
        print("Detections object is missing required attributes.")
        return detections, []

    # Sort by confidence initially
    detections_combined = sorted(zip(detections.confidence, detections.class_id,
                                     detections.xyxy, detections.mask,
                                     range(len(given_labels))),
                                 key=lambda x: x[0],
                                 reverse=True)

    if top_x_detections is not None:
        detections_combined = detections_combined[:top_x_detections]
    print("[before] len(detections_combined):", len(detections_combined))
    # Further filter based on proximity
    filtered_detections = []
    for idx, current_det in enumerate(detections_combined):
        _, curr_class_id, curr_xyxy, curr_mask, _ = current_det
        keep = True

        # Calculate the total number of pixels as a threshold for small masks
        total_pixels = image.shape[0] * image.shape[1]
        small_mask_size = total_pixels * min_mask_size_ratio

        # check mask size and remove if too small
        mask_size = np.count_nonzero(current_det[3])
        if mask_size < small_mask_size:
            print(
                f"Removing {classes.get_classes_arr()[curr_class_id]} because the mask size is too small."
            )
            keep = False

        for other in filtered_detections:
            _, other_class_id, other_xyxy, other_mask, _ = other

            if mask_iou(curr_mask, other_mask) > iou_threshold:
                # 크기가 거의 똑같은 물체 -> 다른 class일 리 없다.
                filtered_detections.remove(other)
                continue
        if classes.get_classes_arr()[curr_class_id] in classes.bg_classes:
            print(
                f"Removing {classes.get_classes_arr()[curr_class_id]} because it is a background class, specifically {classes.bg_classes}."
            )
            keep = True  #False

        if keep:
            filtered_detections.append(current_det)
    print("[after] len(filtered_detections):", len(filtered_detections))
    # Unzip the filtered results
    confidences, class_ids, xyxy, masks, indices = zip(*filtered_detections)
    filtered_labels = [given_labels[i] for i in indices]

    # Create new detections object
    filtered_detections = sv.Detections(class_id=np.array(class_ids,
                                                          dtype=np.int64),
                                        confidence=np.array(confidences,
                                                            dtype=np.float32),
                                        xyxy=np.array(xyxy, dtype=np.float32),
                                        mask=np.array(masks, dtype=np.bool_))

    return filtered_detections, filtered_labels


def get_vlm_annotated_image_path(
    det_exp_vis_path,
    color_path,
    frame_idx=None,
    w_edges=False,
    suffix="annotated_for_vlm.jpg",
):
    # Define suffixes based on whether edges are included
    if w_edges:
        suffix = suffix.replace(".jpg", "_w_edges.jpg")
    if frame_idx is not None:
        return os.path.join(det_exp_vis_path, f"{frame_idx:06d}_{suffix}")

    # Create the file path
    vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(
        ".jpg").with_name((det_exp_vis_path / color_path.name).stem + suffix)
    return str(vis_save_path)


def make_vlm_edges_and_captions(image,
                                curr_det,
                                obj_classes,
                                detection_class_labels,
                                det_exp_vis_path,
                                color_path,
                                make_edges_flag,
                                openai_client,
                                frame_idx=None,
                                save_result=True):
    """
    Process detections by filtering, annotating, and extracting object relationships.

    Args:
        image (numpy.ndarray): The image on which detections are performed.
        curr_det (List): Current detections from the detection model.
        obj_classes (list): Object classes used in detection.
        detection_class_labels (list): Labels for each detection class.
        det_exp_vis_path (str): Directory path for saving visualizations.
        color_path (str): Additional path element for creating unique save paths.
        make_edges_flag (bool): Flag indicating whether to create edges between detected objects.
        openai_client (OpenAIClient): Client object for OpenAI used in relationship extraction.

    Returns:
        tuple: A tuple containing the following elements:
            - detection_class_labels (list): The original labels provided for detection classes.
            - labels (list): The labels after filtering detections.
            - edges (list): List of edges between detected objects if `make_edges_flag` is True, otherwise an empty list.
            - edge_image (numpy.ndarray): Annotated image with edges plotted if `make_edges_flag` is True, otherwise None.
            - captions (list): List of captions for each detected object if `make_edges_flag` is True, otherwise None.
    """
    # Filter the detections
    # labels: detection_class_labels가 필터링된 것: ["sofa chair 0", ...]
    """
    그림 그리고, edge 찾는데에만 필터링 결과가 사용됩니다.
    
- 남은 로직
  - IoU가 80% 이상 겹치면, 신뢰도가 낮은 객체를 제거
  - bg_classes 클래스 제거
    """
    # labels: detection_class_labels가 필터링된 것: ["sofa chair 0", ...]
    if curr_det.xyxy.shape[0] == 0:
        filtered_detections = curr_det
        labels = detection_class_labels
        detection_exists = False
    elif openai_client:
        detection_exists = True
        filtered_detections, labels = filter_detections(
            image=image,  #
            detections=curr_det,
            classes=obj_classes,
            top_x_detections=150000,
            confidence_threshold=0.00001,
            given_labels=detection_class_labels,
        )
    else:
        detection_exists = True
        filtered_detections = curr_det
        labels = detection_class_labels
    edges = []
    captions = []
    edge_image = None
    # vis_save_path_for_vlm
    # room0/exps/s_detections_stride10/vis/frame000000annotated_for_vlm.jpg
    vis_save_path_for_vlm = get_vlm_annotated_image_path(
        det_exp_vis_path, color_path, frame_idx)
    if not detection_exists:
        if save_result:
            cv2.imwrite(str(vis_save_path_for_vlm), image)
        return labels, edges, edge_image, captions
    if make_edges_flag:
        # vis_save_path_for_vlm_edges
        # room0/exps/s_detections_stride10/vis/frame000000annotated_for_vlm_w_edges.jpg
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(
            det_exp_vis_path, color_path, frame_idx, w_edges=True)
        """ annotate_for_vlm
        이미지 위에 탐지된 객체들의 위치와 라벨을 오버레이(overlay)하는 방식으로 이루어짐
        객체에 부여된 라벨들은 숫자와 함께 고유하게 표시되며, 
            이를 OpenAI API에서 객체 관계를 파악하는 데 사용됨
        """
        annotated_image_for_vlm, sorted_indices = annotate_for_vlm(
            image,
            filtered_detections,
            obj_classes,
            labels,
            save_path=vis_save_path_for_vlm)

        label_list = []
        for label in labels:
            label_num = str(label.split(" ")[-1])
            label_name = re.sub(r'\s*\d+$', '', label).strip()
            full_label = f"{label_num}: {label_name}"
            label_list.append(full_label)
        if save_result:
            cv2.imwrite(str(vis_save_path_for_vlm), annotated_image_for_vlm)
        print(f"Line 313, vis_save_path_for_vlm: {vis_save_path_for_vlm}")
        if openai_client:
            # object들에 대한 관계 파악
            # edges = [("1", "on top of", "2"), ("3", "under", "2"), ...]
            edges: List[Tuple[str, str, str]] = get_obj_rel_from_image_gpt4v(
                openai_client, vis_save_path_for_vlm, label_list)
            # object에 대한 captions
            """ captions
[
    {"id": "1", "name": "object1", "caption": "concise description of object1"},
    {"id": "2", "name": "object2", "caption": "concise description of object2"}
]
            """
            captions = get_obj_captions_from_image_gpt4v(
                openai_client, vis_save_path_for_vlm, label_list)
            # 탐지된 객체 간의 관계(엣지)를 이미지에 시각적으로 표시하는 작업입니다.
            if save_result:
                save_path = vis_save_path_for_vlm_edges
            else:
                save_path = None
            edge_image = plot_edges_from_vlm(annotated_image_for_vlm,
                                             edges,
                                             filtered_detections,
                                             obj_classes,
                                             labels,
                                             sorted_indices,
                                             save_path=save_path)
    # labels: detection_class_labels가 필터링된 것: ["sofa chair 0", ...]
    return labels, edges, edge_image, captions


def handle_rerun_saving(use_rerun, save_rerun, exp_suffix, exp_out_path):
    # Save the rerun output if needed
    if use_rerun and save_rerun:
        # rerun_file_path:
        #   exps/r_mapping_stride10/rerun_r_mapping_stride10.rrd
        rerun_file_path = exp_out_path / f"rerun_{exp_suffix}.rrd"
        print("매핑이 완료되었습니다!")
        print("리런 뷰어에서 리런 파일을 저장하려면 지금 저장해야 합니다.")
        print("현재 리런에서 파일을 저장하면서 동시에 로그를 기록하는 것은 불가능합니다.")
        print("또한, 계속 진행하기 전에 뷰어를 닫아 주세요. "
              "이 작업은 많은 RAM을 해제하여 포인트 클라우드를 저장하는 데 도움이 됩니다.")
        print(f"아래의 경로를 복사하여 사용하거나 원하는 경로를 선택해도 됩니다:\n{rerun_file_path}")
        input("그런 다음 계속하려면 Enter 키를 누르세요.")


def measure_time(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        # print(f"Starting {func.__name__}...")
        result = func(
            *args,
            **kwargs)  # Call the function with any arguments it was called with
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds"
        )
        return result  # Return the result of the function call

    return wrapper


def get_exp_config_save_path(exp_out_path, is_detection_config=False):
    params_file_name = "config_params"
    if is_detection_config:
        params_file_name += "_detections"
    return exp_out_path / f"{params_file_name}.json"


def save_hydra_config(hydra_cfg, exp_out_path, is_detection_config=False):
    exp_out_path.mkdir(exist_ok=True, parents=True)
    with open(get_exp_config_save_path(exp_out_path, is_detection_config),
              "w") as f:
        dict_to_dump = cfg_to_dict(hydra_cfg)
        json.dump(dict_to_dump, f, indent=2)


def load_saved_hydra_json_config(exp_out_path):
    with open(get_exp_config_save_path(exp_out_path), "r") as f:
        return json.load(f)


def prepare_detection_paths(dataset_root, scene_id, detections_exp_suffix,
                            force_detection, output_base_path):
    """
    Prepare and return paths needed for detection output, creating directories as needed.
    """
    det_exp_path = get_exp_out_path(dataset_root, scene_id,
                                    detections_exp_suffix)
    if force_detection:
        det_vis_folder_path = get_vis_out_path(det_exp_path)
        det_detections_folder_path = get_det_out_path(det_exp_path)
        os.makedirs(det_vis_folder_path, exist_ok=True)
        os.makedirs(det_detections_folder_path, exist_ok=True)
        return det_exp_path, det_vis_folder_path, det_detections_folder_path
    return det_exp_path


def should_exit_early(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Check if we should exit early
        if data.get("exit_early", False):
            # Reset the exit_early flag to False
            data["exit_early"] = False
            # Write the updated data back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return True
        else:
            return False
    except Exception as e:
        # If there's an error reading the file or the key doesn't exist,
        # log the error and return False
        print(f"Error reading {file_path}: {e}")
        logging.info(f"Error reading {file_path}: {e}")
        return False


def save_detection_results(base_path, results):
    base_path.mkdir(exist_ok=True, parents=True)
    for key, value in results.items():
        save_path = Path(base_path) / f"{key}"
        if isinstance(value, np.ndarray):
            # Save NumPy arrays using .npz for efficient storage
            np.savez_compressed(f"{save_path}.npz", value)
        else:
            # For other types, fall back to pickle
            with gzip.open(f"{save_path}.pkl.gz", "wb") as f:
                pickle.dump(value, f)


def load_saved_detections(base_path):
    base_path = Path(base_path)

    # Construct potential .pkl.gz file path based on the base_path
    potential_pkl_gz_path = Path(str(base_path) + '.pkl.gz')

    # Check if the constructed .pkl.gz file exists
    # This is the old wat
    if potential_pkl_gz_path.exists() and potential_pkl_gz_path.is_file():
        # The path points directly to a .pkl.gz file
        with gzip.open(potential_pkl_gz_path, "rb") as f:
            return pickle.load(f)
    elif base_path.is_dir():
        loaded_detections = {}
        for file_path in base_path.iterdir():
            # Handle files based on their extension, adjusting the key extraction method
            if file_path.suffix == '.npz':
                key = file_path.name.replace('.npz', '')
                with np.load(file_path, allow_pickle=True) as data:
                    loaded_detections[key] = data['arr_0']
            elif file_path.suffix == '.gz' and file_path.suffixes[-2] == '.pkl':
                key = file_path.name.replace('.pkl.gz', '')
                with gzip.open(file_path, "rb") as f:
                    loaded_detections[key] = pickle.load(f)
        return loaded_detections
    else:
        raise FileNotFoundError(
            f"No valid file or directory found at {base_path}")


class ObjectClasses:
    """
객체 클래스 및 그에 대한 색상 정보를 관리하는 역할을 수행
    특정 설정에 따라 배경 클래스를 포함하거나 제외할 수 있음
주로 이미지 분할이나 객체 인식 모델에서 객체별 색상 매핑을 관리하는 데 사용

### 클래스 초기화 (`__init__`):
  - `classes_file_path`:
        클래스 이름들이 정의된 파일의 경로입니다. 이 파일은 각 클래스 이름이 한 줄씩 포함
  - `bg_classes`:
        배경으로 간주될 클래스들의 목록
        기본적으로는 `["wall", "floor", "ceiling"]`와 같은 클래스가 배경으로 간주
  - `skip_bg`:
        배경 클래스를 무시할지 여부를 결정하는 플래그
        `True`일 경우 배경 클래스를 제외하고 나머지 클래스들만 사용
- 클래스 초기화 시 `_load_or_create_colors` 메서드를 호출하여
    클래스 이름과 각 클래스에 대응하는 색상을 초기화

### 색상 로딩 또는 생성 (`_load_or_create_colors`):
- 클래스 이름을 **파일로부터 읽어온 뒤**,
    `skip_bg` 플래그가 `True`일 경우 배경 클래스를 제외한 나머지 클래스를 필터링
- **색상 정보**는 클래스 파일 경로와 동일한 경로에 위치한 `<파일명>_colors.json`에서 로드되며,
    이 파일이 존재할 경우 그 데이터를 사용합니다.
  - 색상 파일이 존재하면, 그 파일에서 클래스 이름에 대응하는 색상 맵을 로드합니다.
    하지만, 클래스 파일의 내용과 색상 파일의 클래스가 불일치할 수 있으므로,
        현재 사용 중인 클래스에 대한 색상만 필터링하여 유지
- **색상 파일이 존재하지 않는 경우**,
    각 클래스에 대해 RGB 값을 난수로 생성한 후 해당 색상 맵을 새로 생성된 JSON 파일에 저장
- 결과적으로, 클래스 이름 리스트와 그에 대응하는 색상 딕셔너리를 반환합니다.

### 클래스 배열 반환 (`get_classes_arr`):
- 현재 관리되고 있는 클래스 이름 리스트를 반환합니다. 이는 배경 클래스가 제외된 상태일 수도 있음

### 배경 클래스 배열 반환 (`get_bg_classes_arr`):
- 배경 클래스로 지정된 클래스들의 리스트를 반환합니다.

### 클래스 색상 반환 (`get_class_color`):
- **클래스 이름 또는 인덱스를 입력**으로 받아 해당 클래스의 색상을 반환합니다.
  - 인덱스가 주어지면, 클래스 리스트에서 해당 인덱스에 해당하는 클래스 이름을 찾아 색상을 반환합니다.
  - 이름이 주어지면, 해당 이름이 클래스 리스트에 있는지 확인하고 그 클래스의 색상을 반환합니다.
  - 색상이 정의되지 않은 경우에는 기본적으로 `[0, 0, 0]` (검정색)을 반환합니다.

### 인덱스별 클래스 색상 딕셔너리 반환 (`get_class_color_dict_by_index`):
- **클래스 인덱스를 키로 하고** 그에 대응하는 색상을 값으로 가지는 딕셔너리를 생성하여 반환
이 과정에서는 `get_class_color` 메서드를 이용하여 각 인덱스의 색상을 가져옵니다.

    Usage:
        obj_classes = ObjectClasses(classes_file_path, skip_bg=True)
        model.set_classes(obj_classes.get_classes_arr())
        some_class_color = obj_classes.get_class_color(index or class_name)
    """

    def __init__(self, classes_file_path, bg_classes, skip_bg):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        self.skip_bg = skip_bg
        self.classes, self.class_to_color = self._load_or_create_colors()

    def _load_or_create_colors(
            self) -> Tuple[List[str], Dict[str, List[float]]]:
        with open(self.classes_file_path, "r") as f:
            all_classes = [cls.strip() for cls in f.readlines()]

        # Filter classes based on the skip_bg parameter
        if self.skip_bg:
            classes = [cls for cls in all_classes if cls not in self.bg_classes]
        else:
            classes = all_classes

        colors_file_path = self.classes_file_path.parent / f"{self.classes_file_path.stem}_colors.json"
        if colors_file_path.exists():
            with open(colors_file_path, "r") as f:
                class_to_color = json.load(f)
            # Ensure color map only includes relevant classes
            class_to_color = {
                cls: class_to_color[cls]
                for cls in classes
                if cls in class_to_color
            }
        else:
            class_to_color = {
                class_name: list(np.random.rand(3).tolist())
                for class_name in classes
            }
            with open(colors_file_path, "w") as f:
                json.dump(class_to_color, f)

        return classes, class_to_color

    def get_classes_arr(self) -> List[str]:
        """
        Returns the list of class names,
        excluding background classes if configured to do so.
        """
        return self.classes

    def get_bg_classes_arr(self):
        """
        Returns the list of background class names, if configured to do so.
        """
        return self.bg_classes

    def get_class_color(self, key):
        """
        Retrieves the color associated with a given class name or index.
        
        Args:
            key (int or str): The index or name of the class.
        
        Returns:
            list: The color (RGB values) associated with the class.
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self.classes):
                raise IndexError("Class index out of range.")
            class_name = self.classes[key]
        elif isinstance(key, str):
            class_name = key
            if class_name not in self.classes:
                raise ValueError(f"{class_name} is not a valid class name.")
        else:
            raise ValueError(
                "Key must be an integer index or a string class name.")
        return self.class_to_color.get(
            class_name, [0, 0, 0])  # Default color for undefined classes

    def get_class_color_dict_by_index(self):
        """
        Returns a dictionary of class colors, just like self.class_to_color, but indexed by class index.
        """
        return {
            str(i): self.get_class_color(i) for i in range(len(self.classes))
        }


def save_obj_json(exp_suffix, exp_out_path, objects):
    """
    Saves the objects to a JSON file with the specified suffix.

    Args:
    - exp_suffix (str): Suffix for the experiment, used in naming the saved file.
    - exp_out_path (Path or str): Output path for the experiment's saved files.
    - objects: The objects to save, assumed to have necessary attributes.
    """
    json_obj_list = {}
    for curr_idx, curr_obj in enumerate(objects):
        obj_key = f"object_{curr_idx + 1}"
        bbox_extent = [round(val, 2) for val in curr_obj['bbox'].extent
                      ]  # Round values to 2 decimal places
        bbox_center = [round(val, 2) for val in curr_obj['bbox'].center
                      ]  # Assuming `center` is an iterable like a list or tuple
        bbox_volume = round(bbox_extent[0] * bbox_extent[1] * bbox_extent[2],
                            2)  # Calculate volume and round to 2 decimal places

        obj_dict = {
            "id": curr_obj['curr_obj_num'],
            "object_tag": curr_obj['class_name'],
            "object_caption": curr_obj['consolidated_caption'],
            "bbox_extent": bbox_extent,
            "bbox_center": bbox_center,
            "bbox_volume": bbox_volume  # Add the volume to the dictionary
        }
        json_obj_list[obj_key] = obj_dict

    json_obj_out_path = Path(exp_out_path) / f"obj_json_{exp_suffix}.json"
    json_obj_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_obj_out_path, "w") as f:
        json.dump(json_obj_list, f, indent=2)
    print(f"Saved object JSON to {json_obj_out_path}")


def save_edge_json(exp_suffix, exp_out_path, objects, edges):
    """
    Saves the edges to a JSON file with the specified suffix.

    Args:
    - exp_suffix (str): Suffix for the experiment, used in naming the saved file.
    - exp_out_path (Path or str): Output path for the experiment's saved files.
    - objects: The objects involved in the edges.
    - edges: The edges to save, assumed to have necessary attributes.
    """
    json_edge_list = {}
    for curr_idx, curr_edge_item in enumerate(list(
            edges.edges_by_index.items())):
        curr_edj_tup, curr_edge = curr_edge_item
        obj1_idx = curr_edge.obj1_idx
        obj2_idx = curr_edge.obj2_idx
        rel_type = curr_edge.rel_type
        num_det = curr_edge.num_detections
        obj1_class_name = objects[obj1_idx]['class_name']
        obj2_class_name = objects[obj2_idx]['class_name']
        obj1_curr_obj_num = objects[obj1_idx]['curr_obj_num']
        obj2_curr_obj_num = objects[obj2_idx]['curr_obj_num']
        # print(f"Line 732, {obj1_class_name} {rel_type} {obj2_class_name}, num_det: {num_det}")

        edj_dict = {
            "edge_id":
                curr_idx,
            "edge_description":
                f"{obj1_class_name} {rel_type} {obj2_class_name}",
            "num_detections":
                num_det,
            "object_1_id":
                obj1_curr_obj_num,
            "object_1_tag":
                obj1_class_name,
            "object_2_id":
                obj2_curr_obj_num,
            "object_2_tag":
                obj2_class_name,
            "relationship":
                rel_type,
        }
        json_edge_list[f"edge_{curr_idx}"] = edj_dict

    json_edge_out_path = Path(exp_out_path) / f"edge_json_{exp_suffix}.json"
    json_edge_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_edge_out_path, "w") as f:
        json.dump(json_edge_list, f, indent=2)
    print(f"Saved edge JSON to {json_edge_out_path}")


def save_pointcloud(exp_suffix,
                    exp_out_path,
                    cfg,
                    objects,
                    obj_classes,
                    latest_pcd_filepath=None,
                    create_symlink=True,
                    edges=None):
    """
객체와 관련된 포인트 클라우드 데이터를 저장하고,
    필요한 경우 최신 파일을 가리키는 심볼릭 링크를 생성 또는 업데이트

### 알고리즘 설명

1. **저장할 결과 딕셔너리 준비**
 - `objects`: 객체 리스트로, `to_serializable()` 메서드를 사용해 직렬화 가능한 형태로 변환
 - `cfg`: 실험 설정 정보를 `cfg_to_dict(cfg)` 함수를 통해 딕셔너리로 변환
 - `class_names`: 객체 클래스 이름 배열로, `get_classes_arr()` 메서드를 통해 가져옵니다.
 - `class_colors`: 객체 클래스의 색상 정보를 `get_class_color_dict_by_index()` 메서드를 통해 가져옵니다.
 - `edges`: 객체 간 연결 정보를 직렬화 가능한 형태로 변환하여 포함시키며, `edges`가 제공되지 않은 경우 `None`으로 설정

2. **저장 경로 설정**
   - 저장 경로는 `exp_out_path`와 `exp_suffix`를 결합하여 설정되며, `.pkl.gz` 형식으로 저장
    """
    print("saving map...")
    # Prepare the results dictionary
    results = {
        'objects': objects.to_serializable(),
        'cfg': cfg_to_dict(cfg),
        'class_names': obj_classes.get_classes_arr(),
        'class_colors': obj_classes.get_class_color_dict_by_index(),
        'edges': edges.to_serializable() if edges is not None else None,
    }

    # Define the save path for the point cloud
    pcd_save_path = Path(exp_out_path) / f"pcd_{exp_suffix}.pkl.gz"
    # Make the directory if it doesn't exist
    pcd_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the point cloud data
    with gzip.open(pcd_save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved point cloud to {pcd_save_path}")
    if edges is not None:
        print(f"Also saved edges to {pcd_save_path}")

    # Create or update the symlink if requested
    if create_symlink and latest_pcd_filepath:
        latest_pcd_path = Path(latest_pcd_filepath)
        # Remove the existing symlink if it exists
        if latest_pcd_path.is_symlink() or latest_pcd_path.exists():
            latest_pcd_path.unlink()
        # Create a new symlink pointing to the latest point cloud save
        latest_pcd_path.symlink_to(pcd_save_path)
        print(
            f"Updated symlink to point to the latest point cloud save at {latest_pcd_path} to:\n{pcd_save_path}"
        )


def find_existing_image_path(base_path, extensions):
    """
    Checks for the existence of a file with the given base path and any of the provided extensions.
    Returns the path of the first existing file found or None if no file is found.

    Parameters:
    - base_path: The base file path without the extension.
    - extensions: A list of file extensions to check for.

    Returns:
    - Path of the existing file or None if no file exists.
    """
    for ext in extensions:
        potential_path = base_path.with_suffix(ext)
        if potential_path.exists():
            return potential_path
    return None


def save_objects_for_frame(obj_all_frames_out_path, frame_idx, objects,
                           obj_min_detections, adjusted_pose, color_path):
    """
이 코드는 주어진 프레임에서 객체 데이터에 대한 정보를 압축된 파일로 저장하는 과정
이 과정은 특정 프레임에서 감지된 객체들을 처리하고, 나중에 다시 사용할 수 있도록 저장하는 데 목적

### 알고리즘 설명
     {obj_all_frames_out_path}/{frame_idx:06d}.pkl.gz # 파일을 압축하여 저장
3. ** 객체 정보를 저장하기에 적합한 형태로 가공**
   - 필터링된 객체 리스트 `filtered_objects`를 `MapObjectList` 형식으로 변환한 후,
        시각화 및 저장 준비가 완료된 형태로 변환하는 `prepare_objects_save_vis` 함수를 호출하여
        `prepared_objects`를 생성

4. **저장할 데이터 구성**
   - 최종적으로 저장할 데이터를 하나의 딕셔너리 형태로 구성
     - `camera_pose`: 조정된 카메라의 위치 정보(`adjusted_pose`).
     - `objects`: 필터링 및 준비된 객체 리스트(`prepared_objects`).
     - `frame_idx`: 현재 프레임 인덱스(`frame_idx`).
     - `num_objects`: 객체의 개수.
     - `color_path`: 프레임에 대한 이미지 경로(`color_path`).

5. **데이터 압축 및 저장**
   -  `pickle`을 사용하여 데이터를 직렬화한 후, `gzip`으로 파일을 압축해 저장
    """
    save_path = obj_all_frames_out_path / f"{frame_idx:06d}.pkl.gz"
    filtered_objects = [
        obj for obj in objects if obj['num_detections'] >= obj_min_detections
    ]
    prepared_objects = prepare_objects_save_vis(MapObjectList(filtered_objects))
    result = {
        "camera_pose": adjusted_pose,
        "objects": prepared_objects,
        "frame_idx": frame_idx,
        "num_objects": len(filtered_objects),
        "color_path": str(color_path)
    }
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(result, f)


def add_info_to_image(image, frame_idx, num_objects, color_path):
    frame_info_text = f"Frame: {frame_idx}, Objects: {num_objects}, Path: {str(color_path)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 0, 0)
    thickness = 1
    line_type = cv2.LINE_AA
    position = (10, image.shape[0] - 10)
    cv2.putText(image, frame_info_text, position, font, font_scale, color,
                thickness, line_type)


def save_video_from_frames(frames, exp_out_path, exp_suffix):
    video_save_path = exp_out_path / (f"s_mapping_{exp_suffix}.mp4")
    save_video_from_frames(frames, video_save_path, fps=10)
    print(f"Save video to {video_save_path}")


def vis_render_image(objects, obj_classes, obj_renderer, image_original_pil,
                     adjusted_pose, frames, frame_idx, color_path,
                     obj_min_detections, class_agnostic, debug_render,
                     is_final_frame, exp_out_path, exp_suffix):
    filtered_objects = [
        deepcopy(obj) for obj in objects if
        obj['num_detections'] >= obj_min_detections and not obj['is_background']
    ]
    objects_vis = MapObjectList(filtered_objects)

    if class_agnostic:
        objects_vis.color_by_instance()
    else:
        objects_vis.color_by_most_common_classes(obj_classes)

    rendered_image, vis = obj_renderer.step(
        image=image_original_pil,
        gt_pose=adjusted_pose,
        new_objects=objects_vis,
        paint_new_objects=False,
        return_vis_handle=debug_render,
    )

    if rendered_image is not None:
        add_info_to_image(rendered_image, frame_idx, len(filtered_objects),
                          color_path)
        frames.append((rendered_image * 255).astype(np.uint8))

    if is_final_frame:
        # Save the video
        save_video_from_frames(frames, exp_out_path, exp_suffix)
