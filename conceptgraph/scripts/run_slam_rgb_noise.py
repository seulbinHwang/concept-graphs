import argparse
import json
import os
# 환경 변수 설정
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from pathlib import Path
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from tqdm import trange

import open3d as o3d

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils import general_utils
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help=
        "This path may need to be changed depending on where you run this script. "
    )
    parser.add_argument("--scene_id", type=str, default="train_3")
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=640)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_pcd", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--load_semseg",
        action="store_true",
        help="Load GT semantic segmentation and run fusion on them. ")

    return parser



def main(args: argparse.Namespace):
    """

export REPLICA_ROOT= ~/PycharmProjects/concept-graphs/Datasets/Replica

export CG_FOLDER=/path/to/concept-graphs/
export REPLICA_CONFIG_PATH=${CG_FOLDER}/conceptgraph/dataset/
    dataconfigs/replica/replica.yaml

source env_vars.bash
SCENE_NAME=room0
python -m conceptgraph.scripts.run_slam_rgb \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 480 \
    --image_width 640 \
    --stride 5 \
    --visualize

dataset_root : $REPLICA_ROOT
    ~/PycharmProjects/concept-graphs/Datasets/Replica
dataset_config : $REPLICA_CONFIG_PATH
    ${CG_FOLDER}/conceptgraph/dataset/dataconfigs/replica/replica.yaml
CG_FOLDER : ~/PycharmProjects/concept-graphs
scene_id : $SCENE_NAME
    room0
    """
    if args.load_semseg:
        load_embeddings = True
        embedding_dir = "embed_semseg"
        # dataset_root: $REPLICA_ROOT = /path/to/Replica
        semseg_classes = json.load(
            open(
                args.dataset_root / args.scene_id / "embed_semseg_classes.json",
                "r"))
        embedding_dim = len(semseg_classes)
    else:
        load_embeddings = False
        embedding_dir = "embeddings"
        embedding_dim = 512
    """
        dataconfig: $REPLICA_CONFIG_PATH
                        ${CG_FOLDER}/conceptgraph/dataset/
                        dataconfigs/replica/replica.yaml
        basedir: $REPLICA_ROOT
                    /path/to/Replica
        sequence: $SCENE_NAME
                    room0
        **kwargs:
    """
    # dataset: ReplicaDataset
    dataset = get_dataset(
        dataconfig=args.dataset_config,
        basedir=args.dataset_root,
        sequence=args.scene_id,
        desired_height=args.image_height,
        desired_width=args.image_width,
        start=args.start,
        end=args.end,
        stride=args.stride,
        load_embeddings=load_embeddings,
        embedding_dir=embedding_dir,
        embedding_dim=embedding_dim,
        relative_pose=False,
    )
    """ RGBDImages
    - rgb / depth / intrinsics / extinsics(pose) / 
    - vertex_map : 로컬 좌표계에서 깊이 맵을 3D 포인트 클라우드로
    - normal_map
    - global_vertex_map / global_normal_map
    - valid_depth_mask
    """


    #     rgbdimages = RGBDImages(colors, depths, intrinsics, poses, channels_first=False)
    slam = PointFusion(odom="gradicp", dsratio=4, device=args.device)
    #     pointclouds, recovered_poses = slam(rgbdimages)

    """ Pointclouds
    - points / normals / colors / confidences 
    - 변환 / 스케일링 / 오프셋 추가 / 핀홀 프로젝션
    -  포인트 클라우드 데이터를 두 가지 형태로 저장할 수 있습니다:
      - 리스트 형태 / 패딩된 형태
    
    """
    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(device=args.device)
    """
    len(dataset): RGB images path에 저장된 장수 (stride로 설정한 값만큼)
    """
    # 랜덤 넘버 생성기의 시드를 설정하여 재현성 확보
    np.random.seed(42)
    ##########################################
    # 최대 노이즈 수준을 변수로 정의합니다.
    max_xyz_noise = 0.1  # 위치 노이즈 최대값 (단위: 미터)
    max_angle_noise_deg = 10  # 각도 노이즈 최대값 (단위: 도)
    max_angle_noise_rad = np.deg2rad(max_angle_noise_deg)
    # 표준편차 계산 (95% 확률로 노이즈가 범위 내에 있음)
    xyz_noise_std = max_xyz_noise / 1.96
    angle_noise_std_deg = max_angle_noise_deg / 1.96
    angle_noise_std_rad = np.deg2rad(angle_noise_std_deg)
    ##########################################
    avg_fixed_minus_gt_deg = np.zeros(6)
    avg_noise_deg = np.zeros(6)
    for idx in trange(len(dataset)):
        if load_embeddings:
            _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
            _embedding = _embedding.unsqueeze(0).half()
            _confidence = torch.ones_like(_embedding)
        else:
            _color, _depth, intrinsics, _pose = dataset[idx]
            """
            _color: (480, 640, 3)
            _depth: (480, 640, 1)
            intrinsics: (4, 4)
            _pose: (4, 4)
            """
            _embedding = None
            _confidence = None

        pose_np_gt = _pose.cpu().numpy()
        pose_flat_deg_np_gt = general_utils.extract_xyz_rpw(pose_np_gt)
        pose_flat_rad_np_gt = pose_flat_deg_np_gt.copy()
        pose_flat_rad_np_gt[3:] = np.deg2rad(pose_flat_deg_np_gt[3:])

        ############### 노이즈 추가 ###############
        # 위치 노이즈 생성 (가우시안 분포, 평균 0, 표준편차 xyz_noise_std)
        xyz_noise = np.random.normal(0, xyz_noise_std, size=3)

        # 각도 노이즈 생성 (가우시안 분포, 평균 0, 표준편차 angle_noise_std_rad)
        rpy_noise = np.random.normal(0, angle_noise_std_rad, size=3)
        rpy_noise_deg = np.rad2deg(rpy_noise)
        noise_flat_deg = np.abs(np.concatenate([xyz_noise, rpy_noise_deg]))

        avg_noise_deg += noise_flat_deg

        # 노이즈를 pose_flat_np_gt에 주입
        pose_flat_rad_np_noisy = pose_flat_rad_np_gt.copy()  # 원본 데이터를 보존하기 위해 복사
        pose_flat_rad_np_noisy[:3] += xyz_noise  # x, y, z에 노이즈 추가
        pose_flat_rad_np_noisy[3:] += rpy_noise  # roll, pitch, yaw에 노이즈 추가
        pose_flat_deg_np_noisy = pose_flat_rad_np_noisy.copy()
        pose_flat_deg_np_noisy[3:] = np.rad2deg(pose_flat_rad_np_noisy[3:])

        ##########################################
        pose_noise_np = general_utils.xyz_rpw_to_transformation_matrix(pose_flat_rad_np_noisy)
        pose_noise_tensor = torch.from_numpy(pose_noise_np).to(_pose.device).to(_pose.dtype)  # (4, 4)




        frame_cur = RGBDImages(
            rgb_image=_color.unsqueeze(0).unsqueeze(0),  # (1, 1, 480, 640, 3)
            depth_image=_depth.unsqueeze(0).unsqueeze(0),  # (1, 1, 480, 640, 1)
            intrinsics=intrinsics.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 4)
            poses=pose_noise_tensor.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 4)
            embeddings=_embedding,  # None
            confidence_image=_confidence,  # None
        )

        pointclouds, recovered_poses = slam.step(pointclouds, live_frame=frame_cur, prev_frame=frame_prev, use_current_pose=True)
        recovered_pose_np = recovered_poses.cpu().numpy().squeeze()
        recovered_pose_flat_deg_np = general_utils.extract_xyz_rpw(recovered_pose_np)
        print("[start]--------------")
        print("pose_flat_deg_np_noisy: ", np.round(pose_flat_deg_np_noisy, 2))
        print("pose_flat_deg_np_gt: ", np.round(pose_flat_deg_np_gt, 2))
        print("recovered_pose_flat_deg_np: ", np.round(recovered_pose_flat_deg_np, 2))
        fixed_minus_gt_deg = np.abs(recovered_pose_flat_deg_np - pose_flat_deg_np_gt)
        avg_fixed_minus_gt_deg += fixed_minus_gt_deg
        print("noise_flat_deg: ", np.round(noise_flat_deg, 2))
        print("fixed_minus_gt_deg: ", np.round(fixed_minus_gt_deg, 2))
        print("[end]--------------")


        frame_prev = frame_cur # Keep it None when we use the gt odom
        torch.cuda.empty_cache()
    avg_fixed_minus_gt_deg /= len(dataset)
    avg_noise_deg /= len(dataset)
    print("-----------------")
    print("avg_fixed_minus_gt_deg: ", np.round(avg_fixed_minus_gt_deg, 2))
    print("avg_noise_deg: ", np.round(avg_noise_deg, 2))
    print("-----------------")
    # dataset_root =  $REPLICA_ROOT = /path/to/Replica
    # scene_id  = $SCENE_NAME = room0
    # dir_to_save_map = /path/to/Replica/room0/rgb_cloud
    dir_to_save_map = os.path.join(args.dataset_root, args.scene_id,
                                   "rgb_cloud")
    print(f"Saving the map to {dir_to_save_map}")
    os.makedirs(dir_to_save_map, exist_ok=True)
    pointclouds.save_to_h5(dir_to_save_map)

    # Set the filename for the PCD file
    # pcd_file_path = /path/to/Replica/room0/rgb_cloud/pointcloud.pcd
    pcd_file_path = os.path.join(dir_to_save_map, "pointcloud.pcd")
    pcd = pointclouds.open3d(index=0)
    print("success to get open3d pointcloud from pointclouds.")

    if args.save_pcd:
        o3d.io.write_point_cloud(pcd_file_path, pcd)  # Saving as PCD
        print(f"Saved pointcloud to {pcd_file_path}")

    if args.visualize:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
