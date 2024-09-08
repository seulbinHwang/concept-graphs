import open3d as o3d
aabb = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [1, 1, 1])
print(aabb.get_center())  # center 확인
print(aabb.get_extent())  # extent 확인
