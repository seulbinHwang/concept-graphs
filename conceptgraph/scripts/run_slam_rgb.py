import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

import open3d as o3d

from conceptgraph.dataset.datasets_common import get_dataset

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

    slam = PointFusion(odom="gt", dsratio=1, device=args.device)

    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(device=args.device)
    """
    len(dataset): RGB images path에 저장된 장수 (stride로 설정한 값만큼)
    """
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

        pose_np = _pose.cpu().numpy()

        _pose = torch.from_numpy(pose_np).to(_pose.device).to(_pose.dtype)

        frame_cur = RGBDImages(
            rgb_image=_color.unsqueeze(0).unsqueeze(0),  # (1, 1, 480, 640, 3)
            depth_image=_depth.unsqueeze(0).unsqueeze(0),  # (1, 1, 480, 640, 1)
            intrinsics=intrinsics.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 4)
            poses=_pose.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 4)
            embeddings=_embedding,  # None
            confidence_image=_confidence,  # None
        )

        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev)
        # frame_prev = frame_cur # Keep it None when we use the gt odom
        torch.cuda.empty_cache()
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
