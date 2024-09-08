import open3d as o3d

import numpy as np

# 샘플 포인트 클라우드 생성
pcd = o3d.geometry.PointCloud()

# 랜덤하게 포인트 클라우드 데이터 생성
points = np.random.rand(100, 3)
pcd.points = o3d.utility.Vector3dVector(points)

# OrientedBoundingBox를 robust 모드로 계산
obb = pcd.get_oriented_bounding_box(robust=True)

# 포인트 클라우드와 Bounding Box 시각화
o3d.visualization.draw_geometries([pcd, obb])

# Bounding Box 정보 출력
print("Oriented Bounding Box:")
print(f"Center: {obb.center}")
print(f"Center: {obb.get_center()}")
print(f"Extent: {obb.extent}")
print(f"Extent: {obb.get_extent()}")
print(f"Rotation Matrix: \n{obb.R}")


aabb = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [1, 1, 1])
print(aabb.center)  # center 확인
print(aabb.extent)  # extent 확인
