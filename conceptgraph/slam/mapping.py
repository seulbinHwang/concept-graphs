from conceptgraph.utils.logging_metrics import MappingTracker
import torch
import torch.nn.functional as F

from typing import List, Optional

from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.utils.general_utils import Timer
from conceptgraph.utils.ious import (
    compute_iou_batch,
    compute_giou_batch,
    compute_3d_iou_accurate_batch,
    compute_3d_giou_accurate_batch,
)
from conceptgraph.slam.utils import (compute_overlap_matrix_general,
                                     merge_obj2_into_obj1,
                                     compute_overlap_matrix_2set)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB

owandb = OptionalWandB()

tracker = MappingTracker()


def compute_spatial_similarities(spatial_sim_type: str,
                                 detection_list: DetectionList,
                                 objects: MapObjectList,
                                 downsample_voxel_size) -> torch.Tensor:
    # det_bboxes: (M, 8, 3)
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    # obj_bboxes: (N, 8, 3)
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if spatial_sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    elif spatial_sim_type == "giou":
        spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif spatial_sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accurate_batch(det_bboxes, obj_bboxes)
    elif spatial_sim_type == "giou_accurate":
        spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    elif spatial_sim_type == "overlap":
        # 0.01m 만큼 가까워야, 두 point가 중첩으로 간주
        # spatial_sim: (누적 물체 개수, 새 검지 개수) -> 각 값은 obj와 det의 중첩 비율
        spatial_sim = compute_overlap_matrix_general(objects, detection_list,
                                                     downsample_voxel_size)
        # spatial_sim : (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 중첩 비율
        spatial_sim = torch.from_numpy(spatial_sim).T
    else:
        raise ValueError(f"Invalid spatial similarity type: {spatial_sim_type}")

    return spatial_sim


def compute_visual_similarities(detection_list: DetectionList,
                                objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    det_fts = detection_list.get_stacked_values_torch('clip_ft')  # (M, D)
    obj_fts = objects.get_stacked_values_torch('clip_ft')  # (N, D)

    det_fts = det_fts.unsqueeze(-1)  # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0)  # (1, D, N)

    # (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 코싸인 유사도
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1)  # (M, N)

    return visual_sim


def aggregate_similarities(match_method: str, phys_bias: float,
                           spatial_sim: torch.Tensor,
                           visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
         - (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 중첩 비율
        visual_sim: a MxN tensor of visual similarities
          - (새 검지 개수, 누적 물체 개수) -> 각 값은 det와 obj의 코싸인 유사도
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if match_method == "sim_sum":
        sims = (1 + phys_bias) * spatial_sim + (1 - phys_bias) * visual_sim
    else:
        raise ValueError(f"Unknown matching method: {match_method}")

    return sims


def match_detections_to_objects(
    agg_sim: torch.Tensor, detection_threshold: float = float('-inf')
) -> List[Optional[int]]:
    """
    Matches detections to objects based on similarity, returning match indices or None for unmatched.

    Args:
        agg_sim: Similarity matrix (detections vs. objects).
        detection_threshold: Threshold for a valid match (default: -inf).

    Returns:
        List of matching object indices (or None if unmatched) for each detection.
    """
    match_indices = []
    for detected_obj_idx in range(agg_sim.shape[0]):
        max_sim_value = agg_sim[detected_obj_idx].max()
        if max_sim_value <= detection_threshold:
            match_indices.append(None)
        else:
            match_indices.append(agg_sim[detected_obj_idx].argmax().item())

    return match_indices


def merge_obj_matches(
    detection_list: DetectionList,
    objects: MapObjectList,
    match_indices: List[Optional[int]],
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
) -> MapObjectList:
    """
### 1. **주요 역할**
- **탐지된 객체 리스트**와 **기존 객체 리스트**를 비교해,
    - 두 리스트 간의 객체들을 **병합**하거나 **새로 추가**
- 객체가 병합될 때, 두 객체의 **특성(특징 벡터, 포인트 클라우드 등)**이 결합
- 병합 작업이 끝난 후, 객체의 수와 병합 횟수 같은 정보를 **추적**하고 **기록**

### 2. **세부 알고리즘 로직**

1. **탐지된 객체 처리**:
   - **탐지된 객체 리스트**(`detection_list`)와 **기존 객체 리스트**(`objects`)를 순회하면서,
        - 각각의 객체가 **매칭되는지 여부**를 판단
   - **`match_indices`** 리스트는 각 탐지된 객체가 기존 객체와 매칭되는 인덱스를 제공
     - 매칭되지 않는 경우(`None`), 객체는 **새로운 객체**로 추가
     - 매칭된 경우, 두 객체는 병합

2. **객체 병합 또는 추가**:
   - **매칭되지 않은 객체**는 단순히 **기존 객체 리스트에 추가**
   - **매칭된 객체**는 **`merge_obj2_into_obj1`** 함수를 통해 두 객체를 병합
    - 병합 과정에서는 객체의 **포인트 클라우드**나 **특징 벡터**가 결합되며,
        - DBSCAN 같은 알고리즘을 통해 **노이즈 제거**도 수행할 수 있음

3. **객체 수와 병합 횟수 추적**:
   - **병합된 객체**와 **새로 추가된 객체**의 수는 추적기(`tracker`)를 통해 관리
        - 병합 작업이 끝날 때마다 **업데이트**
   - 병합 횟수나 총 객체 수 등의 통계 정보는 Wandb 같은 기록 도구를 통해 저장

    """
    global tracker
    temp_curr_object_count = tracker.curr_object_count
    for detected_obj_idx, existing_obj_match_idx in enumerate(match_indices):
        if existing_obj_match_idx is None:
            # track the new object detection
            tracker.object_dict.update({
                "id": detection_list[detected_obj_idx]['id'],
                "first_discovered": tracker.curr_frame_idx
            })

            objects.append(detection_list[detected_obj_idx])
        else:

            detected_obj = detection_list[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            merged_obj = merge_obj2_into_obj1(
                obj1=matched_obj,
                obj2=detected_obj,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=dbscan_remove_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
                spatial_sim_type=spatial_sim_type,
                device=device,
                run_dbscan=False,
            )
            objects[existing_obj_match_idx] = merged_obj
    tracker.increment_total_merges(
        len(match_indices) - match_indices.count(None))
    tracker.increment_total_objects(len(objects) - temp_curr_object_count)
    # wandb.log({"merges_this_frame" :len(match_indices) - match_indices.count(None)})
    # wandb.log({"total_merges": tracker.total_merges})
    owandb.log({
        "merges_this_frame": len(match_indices) - match_indices.count(None),
        "total_merges": tracker.total_merges,
        "frame_idx": tracker.curr_frame_idx,
    })
    return objects


def merge_detections_to_objects(downsample_voxel_size: float,
                                dbscan_remove_noise: bool, dbscan_eps: float,
                                dbscan_min_points: int, spatial_sim_type: str,
                                device: str, match_method: str,
                                phys_bias: float, detection_list: DetectionList,
                                objects: MapObjectList,
                                agg_sim: torch.Tensor) -> MapObjectList:
    for detected_obj_idx in range(agg_sim.shape[0]):
        if agg_sim[detected_obj_idx].max() == float('-inf'):
            objects.append(detection_list[detected_obj_idx])
        else:
            existing_obj_match_idx = agg_sim[detected_obj_idx].argmax()
            detected_obj = detection_list[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            merged_obj = merge_obj2_into_obj1(
                obj1=matched_obj,
                obj2=detected_obj,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=dbscan_remove_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
                spatial_sim_type=spatial_sim_type,
                device=device,
                run_dbscan=False)
            objects[existing_obj_match_idx] = merged_obj

    return objects
