import os
import re
from typing import List
from ultralytics import YOLO, SAM
from conceptgraph.utils.general_utils import ObjectClasses
from conceptgraph.utils.vis import vis_result_fast
import torch
import cv2
import numpy as np
import supervision as sv
import time
from tqdm import trange, tqdm
import open_clip
from conceptgraph.utils.model_utils import compute_clip_features_batched


def save_cropped_images(image_crops,
                        folder_path: str) -> None:
    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 각각의 이미지들을 폴더에 저장
    for i, cropped_image in enumerate(image_crops):
        # 이미지 파일 이름을 생성 (예: crop_0.png, crop_1.png ...)
        image_path = os.path.join(folder_path, f"crop_{i}.png")

        # 이미지 저장
        cropped_image.save(image_path)

def get_sorted_image_paths(folder_path: str) -> List[str]:
    """
    Get the list of image file paths from the folder, sorted by the numeric value in the file name.

    :param folder_path: Path to the folder containing image files
    :return: List of sorted file paths
    """
    # Regular expression to find the numeric part in the file name (e.g., frame.1.png -> 1)
    number_pattern = re.compile(r'(\d+)')

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter for .png files and sort by the numeric value in the file name
    sorted_files = sorted(
        [f for f in files if f.endswith('.png')],
        key=lambda x: int(
            number_pattern.search(x).group(0)) if number_pattern.search(
            x) else float('inf')
    )

    # Create the full path for each sorted file
    sorted_paths = [os.path.join(folder_path, f) for f in sorted_files]

    return sorted_paths

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    classes_file = "./conceptgraph/scannet200_classes.txt"
    obj_classes = ObjectClasses(
        classes_file_path=classes_file,
        bg_classes=["wall", "floor", "ceiling"], # "wall", "floor", "ceiling"
        skip_bg=False) # False
    # s , m, l , x  //// world / world-v2
    detection_model = YOLO('yolov8l-world.pt')
    detection_model.set_classes(obj_classes.get_classes_arr())
    save_dir = "yolo_world_sam"
    os.makedirs(save_dir, exist_ok=True)

    sam_predictor = SAM('sam_b.pt')  # SAM('mobile_sam.pt') # UltraLytics SAM

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k")
    clip_model = clip_model.to("cpu")
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    folder_path = 'frames_resized'  # Replace with the folder containing image files
    sorted_paths = get_sorted_image_paths(folder_path)
    elapsed_time = 0.
    try:
        for frame_idx, color_path in tqdm(enumerate(sorted_paths),
                                          total=len(sorted_paths)):
            if frame_idx > 15:
                break
            start_time = time.time()
            image = cv2.imread(str(color_path))  # This will in BGR color space
            results = detection_model.predict(color_path,
                                              conf=0.25,
                                              verbose=False,
                                              device=device)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(
                int)  # (N,)

            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = np.round(xyxy_tensor.cpu().numpy(), 2)
            if xyxy_tensor.numel() != 0:
                # segmentation
                sam_out = sam_predictor.predict(color_path,
                                                bboxes=xyxy_tensor,
                                                verbose=False,
                                                device=device)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()  # (N, H, W)
            else:
                masks_np = np.empty((0, *image.shape[:2]),
                                    dtype=np.float64)

            curr_det = sv.Detections(
                xyxy=xyxy_np.copy(),
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            vis_save_path = os.path.join(save_dir, f"{frame_idx}.jpg")
            a_elapsed_time = time.time() - start_time
            elapsed_time += a_elapsed_time
            # annotated_image, labels = vis_result_fast(
            #     image, curr_det, obj_classes.get_classes_arr(), draw_bbox=True)
            # cv2.imwrite(str(vis_save_path), annotated_image)
            image = cv2.imread(str(color_path))  # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (H, W, 3)

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess,
                clip_tokenizer, obj_classes.get_classes_arr(), "cpu")
            save_cropped_images(image_crops, "test")
            raise ValueError("Stop here")
    except:
        pass
    finally:
        elapsed_time_per_frame = elapsed_time / (frame_idx+1)
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"Elapsed time per frame: {elapsed_time_per_frame} seconds")
