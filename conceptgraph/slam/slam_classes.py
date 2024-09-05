from collections.abc import Iterable
import copy
import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)


class DetectionList(list):
    """
이 클래스는 **`DetectionList`**라는 커스텀 리스트 클래스로, **객체 탐지 결과**를 효율적으로 관리하고 처리하는 기능을 제공합니다. 기본적인 리스트 기능을 확장하여, 각 객체 탐지 항목에 대한 다양한 연산과 시각적 처리를 지원하는 메서드를 추가한 것입니다.

### 1. **주요 역할**
- **탐지된 객체 데이터**를 리스트 형태로 관리하면서, 각 객체에 대해 **속성 값 추출**, **값 변환**, **색상 처리** 등을 수행합니다.
- **객체 탐지 결과**를 처리하기 위한 **여러 가지 도구**를 제공합니다. 예를 들어, 각 탐지 객체의 속성 값을 추출하고, 이를 **배열 형태**(PyTorch 텐서 또는 Numpy 배열)로 변환하거나, 탐지된 객체를 **시각적으로 구별**할 수 있도록 색상을 적용할 수 있습니다.

### 2. **세부 알고리즘 로직**

1. **`get_values`**:
   - 각 탐지 객체에서 특정 속성 값을 추출합니다.
   - 인덱스가 주어지면 해당 인덱스의 값만 반환하고, 그렇지 않으면 전체 값을 반환합니다.

2. **`get_stacked_values_torch`**:
   - 각 객체의 속성 값을 추출하고, 이를 **PyTorch 텐서**로 변환한 후, **배치(batch)** 형태로 **스택(stack)**합니다.
   - 속성 값이 **3D 바운딩 박스**일 경우, 그 값을 배열로 변환한 후 텐서로 변환합니다.

3. **`get_stacked_values_numpy`**:
   - `get_stacked_values_torch`에서 반환된 텐서를 **Numpy 배열**로 변환합니다.

4. **리스트 확장 연산 (`__add__`, `__iadd__`)**:
   - 두 개의 `DetectionList` 객체를 합쳐서 새로운 리스트를 반환하거나, 현재 리스트에 새로운 항목을 추가합니다.

5. **부분 리스트 추출** (`slice_by_indices`, `slice_by_mask`):
   - 주어진 **인덱스** 또는 **마스크**에 따라 리스트의 일부만 추출하여 새로운 리스트를 생성합니다.

6. **`get_most_common_class`**:
   - 각 탐지 객체에서 가장 많이 등장하는 **클래스**를 찾습니다. 예를 들어, 객체가 여러 클래스에 속할 수 있을 때 가장 빈번한 클래스를 반환합니다.

7. **`color_by_most_common_classes`**:
   - 탐지된 객체를 가장 많이 등장하는 클래스의 색상으로 **포인트 클라우드**와 **바운딩 박스**를 색칠합니다.
   - 객체의 클래스에 따라 시각적으로 구분하기 위한 처리입니다.

8. **`color_by_instance`**:
   - 각 탐지 객체에 **고유한 색상**을 부여하여 시각적 구분을 합니다.
   - 탐지 객체에 이미 색상이 있으면 이를 사용하고, 없으면 **색상 맵**을 이용해 자동으로 색상을 생성해 적용합니다.

### 3. **결론**
`DetectionList` 클래스는 탐지된 객체의 데이터를 효율적으로 관리하고, 각 객체의 속성 값을 **추출**, **변환**, **시각적 구분**을 제공하는 여러 기능을 추가한 확장 리스트입니다. 객체 간의 구분을 돕기 위해 **클래스별 색상 적용**이나 **객체별 색상 구분** 같은 시각적 처리를 쉽게 할 수 있게 설계되었습니다.

    """
    def get_values(self, key, idx: int = None):
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]

    def get_stacked_values_torch(self, key, idx: int = None):
        """
**리스트 형태로 저장된 객체들**에서 특정한 **속성 값**(특히, 텐서로 변환 가능한 값)을 추출하고,
이를 **PyTorch 텐서**로 변환해 **배치(batch) 형태로 스택**하는 역할
    - 입력된 값이 **3D 바운딩 박스**일 경우,
        - 이를 **Numpy 배열**로 변환한 후 **텐서**로 변환하는 작업을 수행

### 2. **세부 로직**
1. **객체 속성 값 추출**:
   - 각 객체(`detection`)에서 **`key`에 해당하는 값**을 가져옴
   - 만약 **인덱스(`idx`)**가 주어졌다면, 해당 인덱스에 맞는 값을 선택

3. **Numpy 배열을 텐서로 변환**:
   - 값이 **Numpy 배열**일 경우, 이를 **PyTorch 텐서**로 변환합니다.

4. **텐서 스택**:
   - 각 객체의 텐서를 모아서 **배치 형태**로 스택한 후, 이를 반환

        """
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points()) # (8, 3)
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)

    def get_stacked_values_numpy(self, key, idx: int = None):
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)

    def __add__(self, other):
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list

    def __iadd__(self, other):
        self.extend(other)
        return self

    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist of the current list by indexing
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self

    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist of the current list by masking
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self

    def get_most_common_class(self) -> list[int]:
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_id']),
                                       return_counts=True)
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes

    def color_by_most_common_classes(self,
                                     obj_classes,
                                     color_bbox: bool = True):
        '''
        Color the point cloud of each detection by the most common class
        '''
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            # color = obj_classes[str(c)]
            color = obj_classes.get_class_color(int(c))
            d['pcd'].paint_uniform_color(color)
            if color_bbox:
                d['bbox'].color = color

    def color_by_instance(self):
        if len(self) == 0:
            # Do nothing
            return

        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]


class MapObjectList(DetectionList):
    """
- 이 클래스는 **`MapObjectList`**라는 커스텀 리스트로, **`DetectionList`**를 확장한 것
    - 추가 기능 제공
        - **객체 데이터 관리**
        - **유사도 계산**
        - **객체 리스트의 직렬화 및 역직렬화**
- 이 기능들은 객체 감지 후 데이터를 관리하고,
    - 저장하거나 불러오는 데 필요한 작업을 효율적으로 처리할 수 있게 합니다.


### 2. **세부 로직**

1. **유사도 계산 (`compute_similarities`)**:
   - 입력된 **특징 벡터(new_clip_ft)**와 리스트 내의 객체들의 **특징 벡터(clip_ft)** 간의
        - **코사인 유사도**를 계산
   - 입력이 Numpy 배열일 경우, 이를 **PyTorch 텐서**로 변환한 후 유사도를 계산
   - 계산된 유사도는 각 객체와의 유사성을 나타내며, 이후 매칭이나 비교에 사용할 수 있음

2. **객체 직렬화 (`to_serializable`)**:
   - 객체 데이터를 **저장 가능한 형태**로 변환
   - 3D 객체 정보(포인트 클라우드, 경계 상자 등)를 **Numpy 배열**로 변환하고, 이를 딕셔너리 형태로 저장
   - 직렬화된 리스트는 "포인트 클라우드"와 "경계 상자"를 제외한
        - 나머지 정보를 보존하며, 원본 데이터를 복원할 수 있도록 설계

3. **객체 역직렬화 (`load_serializable`)**:
   - 직렬화된 데이터를 사용해 객체 리스트를 **복원**
   - 저장된 Numpy 배열을 **포인트 클라우드**와 **경계 상자**로 변환하고, 이를 원래 형태로 복원
   - 복원된 데이터는 다시 객체 리스트로 추가

    """
    def compute_similarities(self, new_clip_ft):
        '''
        The input feature should be of shape (D, ), a one-row vector
        This is mostly for backward compatibility
        '''
        # if it is a numpy array, make it a tensor
        new_clip_ft = to_tensor(new_clip_ft)

        # assuming cosine similarity for features
        clip_fts = self.get_stacked_values_torch('clip_ft')

        similarities = F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)
        # return similarities.squeeze()
        return similarities

    def to_serializable(self):
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)

            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            # s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])

            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(
                s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)

            del s_obj_dict['pcd']
            del s_obj_dict['bbox']

            s_obj_list.append(s_obj_dict)

        return s_obj_list

    def load_serializable(self, s_obj_list):
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)

            new_obj['clip_ft'] = to_tensor(new_obj['clip_ft'])
            # new_obj['text_ft'] = to_tensor(new_obj['text_ft'])

            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(
                new_obj['pcd_np'])
            new_obj[
                'bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(new_obj['bbox_np']))
            new_obj['bbox'].color = new_obj['pcd_color_np'][0]
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(
                new_obj['pcd_color_np'])

            del new_obj['pcd_np']
            del new_obj['bbox_np']
            del new_obj['pcd_color_np']

            self.append(new_obj)


# not sure if I will use this
class MapEdge():

    def __init__(self,
                 obj1_idx,
                 obj2_idx,
                 rel_type,
                 num_detections=1,
                 first_detected=None):
        self.obj1_idx = obj1_idx
        self.obj2_idx = obj2_idx
        self.rel_type = rel_type
        self.num_detections = num_detections
        self.first_detected = first_detected  # frame index that the object was first detected

    def to_serializable(self):
        return {
            'obj1_idx': self.obj1_idx,
            'obj2_idx': self.obj2_idx,
            'rel_type': self.rel_type,
        }

    def load_serializable(self, s_edge_dict):
        self.obj1_idx = s_edge_dict['obj1_idx']
        self.obj2_idx = s_edge_dict['obj2_idx']
        self.rel_type = s_edge_dict['rel_type']

    def __str__(self):
        return f"({self.obj1_idx}, {self.rel_type}, {self.obj2_idx}), num_det: {self.num_detections}"

    def __repr__(self):
        return str(self)


class MapEdgeMapping:

    def __init__(self, objects):
        self.objects = objects  # Reference to the list of existing objects
        self.edges_by_index = {}  # {(obj1_index, obj2_index): MapEdge}
        self.edges_by_uuid = {}  # {(obj1_uuid, obj2_uuid): MapEdge}

    def add_or_update_edge(self,
                           obj1_index,
                           obj2_index,
                           rel_type,
                           first_detected=None):
        obj1_uuid, obj2_uuid = self.objects[obj1_index]['id'], self.objects[
            obj2_index]['id']
        uuid_key = (obj1_uuid, obj2_uuid)

        if obj1_index == obj2_index:
            print(f"LOOOPY EDGE DETECTED: {obj1_index} == {obj2_index}")
            pass

        if (obj1_index, obj2_index) in self.edges_by_index:
            edge = self.edges_by_index[(obj1_index, obj2_index)]
            edge.num_detections += 1
        else:
            edge = MapEdge(obj1_index,
                           obj2_index,
                           rel_type,
                           first_detected=first_detected)
            self.edges_by_index[(obj1_index, obj2_index)] = edge
            self.edges_by_uuid[uuid_key] = edge

    def delete_edge(self, obj1_index, obj2_index):
        # Check if the edge exists
        if (obj1_index, obj2_index) in self.edges_by_index:
            # Get the UUIDs of the objects
            obj1_uuid = self.objects[obj1_index]['id']
            obj2_uuid = self.objects[obj2_index]['id']
            uuid_key = (obj1_uuid, obj2_uuid)

            # Remove the edge from both index-based and UUID-based dictionaries
            del self.edges_by_index[(obj1_index, obj2_index)]
            if uuid_key in self.edges_by_uuid:
                del self.edges_by_uuid[uuid_key]
            else:
                # If the edge is not found in the UUID-based dictionary, print a warning
                print(
                    f"Edge between {obj1_index} and {obj2_index} not found in UUID-based storage."
                )
            print(
                f"Edge between {obj1_index} and {obj2_index} deleted successfully."
            )
        else:
            print(f"Edge between {obj1_index} and {obj2_index} does not exist.")

    def delete_object_edges(self, obj_index):
        # Remove all edges associated with the object at obj_index
        to_remove = [key for key in self.edges_by_index if obj_index in key]
        for key in to_remove:
            # Remove from both index-based and UUID-based storage
            del self.edges_by_index[key]
            uuid_key = (self.objects[key[0]]['id'], self.objects[key[1]]['id'])
            del self.edges_by_uuid[uuid_key]

    def update_indices(self, index_map, new_objects):
        self.objects = new_objects  # Update the objects reference if necessary
        new_edges_by_index = {}
        new_edges_by_uuid = {}

        for (old_obj1_index,
             old_obj2_index), edge in list(self.edges_by_index.items()):
            new_obj1_index = index_map.get(old_obj1_index)
            new_obj2_index = index_map.get(old_obj2_index)

            if new_obj1_index is not None and new_obj2_index is not None:
                new_key = (new_obj1_index, new_obj2_index)
                new_uuid_key = (self.objects[new_obj1_index]['id'],
                                self.objects[new_obj2_index]['id'])

                if new_key in new_edges_by_index:
                    new_edges_by_index[
                        new_key].num_detections += edge.num_detections
                else:
                    edge.obj1 = new_obj1_index  # Update the edge's internal object index reference
                    edge.obj2 = new_obj2_index
                    new_edges_by_index[new_key] = edge
                    new_edges_by_uuid[new_uuid_key] = edge

        self.edges_by_index = new_edges_by_index
        self.edges_by_uuid = new_edges_by_uuid

    def merge_update_indices(self, index_updates):
        """Update all edge indices based on the new mapping after merging objects."""
        updated_edges_by_index = {}
        updated_edges_by_uuid = {}

        # Iterate over current edges to update indices based on index_updates
        for (obj1_index,
             obj2_index), curr_edge in list(self.edges_by_index.items()):
            new_obj1_index = index_updates[obj1_index]
            new_obj2_index = index_updates[obj2_index]

            # Skip updates if either index is None (meaning the object was merged away)
            if new_obj1_index is None or new_obj2_index is None:
                continue

            # Avoid creating a loop edge where an object points to itself
            if new_obj1_index == new_obj2_index:
                print(
                    f"LOOOPY EDGE DETECTED: {new_obj1_index} == {new_obj2_index}"
                )
                continue

            new_key = (new_obj1_index, new_obj2_index)
            new_obj1_uuid, new_obj2_uuid = self.objects[new_obj1_index][
                'id'], self.objects[new_obj2_index]['id']
            new_uuid_key = (new_obj1_uuid, new_obj2_uuid)

            # If the edge already exists after merge, update num_detections
            if new_key in updated_edges_by_index:
                updated_edges_by_index[
                    new_key].num_detections += curr_edge.num_detections
            else:
                # Update the edge with new indices
                curr_edge.obj1_idx = new_obj1_index
                curr_edge.obj2_idx = new_obj2_index
                updated_edges_by_index[new_key] = curr_edge
                updated_edges_by_uuid[new_uuid_key] = curr_edge

        # Update the class attributes with the modified edges
        self.edges_by_index = updated_edges_by_index
        self.edges_by_uuid = updated_edges_by_uuid

    def update_objects_list(self, new_objects):
        self.objects = new_objects

    def merge_objects_edges(self, source_index, destination_index):
        # Update edges for a merged object. source_index object is merged into destination_index object
        updated_edges_by_index = {}
        updated_edges_by_uuid = {}

        for (obj1_index, obj2_index), curr_edge in self.edges_by_index.items():
            # Check if source object is part of the edge and update the edge accordingly

            # if not (source_index in (obj1_index, obj2_index)):
            #     continue

            new_obj1_index, new_obj2_index = obj1_index, obj2_index

            if new_obj1_index == new_obj2_index:  # check loop edge
                print(
                    f"LOOOPY EDGE DETECTED: {new_obj1_index} == {new_obj2_index}"
                )
                pass

            # check if edge is between source and destination
            if source_index in (new_obj1_index,
                                new_obj2_index) and destination_index in (
                                    new_obj1_index, new_obj2_index):
                print(
                    f"Edge between source and destination: {source_index} in {new_obj1_index, new_obj2_index} and {destination_index} in {new_obj1_index, new_obj2_index}"
                )
                pass
                continue

            if obj1_index == source_index:
                print(
                    f"obj1_index matches source_index: {obj1_index} == {source_index}"
                )
                new_obj1_index = destination_index

            if obj2_index == source_index:
                print(
                    f"obj2_index matches source_index: {obj2_index} == {source_index}"
                )
                new_obj2_index = destination_index

            if new_obj1_index == new_obj2_index:  # check loop edge
                print(
                    f"LOOOPY EDGE DETECTED: {new_obj1_index} == {new_obj2_index}"
                )
                pass
                continue

            # Generate new edge key and UUID key
            new_key = (new_obj1_index, new_obj2_index)
            new_obj1_uuid, new_obj2_uuid = self.objects[new_obj1_index][
                'id'], self.objects[new_obj2_index]['id']
            new_uuid_key = (new_obj1_uuid, new_obj2_uuid)
            new_edge = MapEdge(new_obj1_index, new_obj2_index,
                               curr_edge.rel_type, curr_edge.num_detections)

            # Check if the edge already exists after merge, update num_detections if it does
            if new_key in updated_edges_by_index:
                updated_edges_by_index[
                    new_key].num_detections += curr_edge.num_detections
            else:
                curr_edge.obj1_idx = new_obj1_index
                curr_edge.obj2_idx = new_obj2_index
                updated_edges_by_index[new_key] = new_edge
                updated_edges_by_uuid[new_uuid_key] = new_edge

        # Update the class attributes
        self.edges_by_index = updated_edges_by_index
        self.edges_by_uuid = updated_edges_by_uuid

    def get_edges_by_curr_obj_num(self):
        map_edges_by_curr_obj_num = []
        for (obj1_idx, obj2_idx), map_edge in self.edges_by_index.items():
            obj1_curr_obj_num = self.objects[obj1_idx]['curr_obj_num']
            obj2_curr_obj_num = self.objects[obj2_idx]['curr_obj_num']
            rel_type = map_edge.rel_type
            map_edges_by_curr_obj_num.append(
                (obj1_curr_obj_num, rel_type, obj2_curr_obj_num))
        return map_edges_by_curr_obj_num

    def get_edges_by_curr_obj_num_label(self):
        map_edges_by_curr_obj_num_label = []
        for (obj1_idx, obj2_idx), map_edge in self.edges_by_index.items():
            # Construct the curr_obj_num_label for both objects
            obj1 = self.objects[obj1_idx]
            obj2 = self.objects[obj2_idx]
            obj1_curr_obj_num_label = f"{obj1['curr_obj_num']}_{obj1['class_name']}"
            obj2_curr_obj_num_label = f"{obj2['curr_obj_num']}_{obj2['class_name']}"

            # Append the edge with the formatted labels
            map_edges_by_curr_obj_num_label.append(
                (obj1_curr_obj_num_label, map_edge.rel_type,
                 obj2_curr_obj_num_label))
        return map_edges_by_curr_obj_num_label

    def get_edge_endpoints(self, obj1_index, obj2_index):
        # Check if the edge exists
        if (obj1_index, obj2_index) in self.edges_by_index:
            obj1_center = np.asarray(
                self.objects[obj1_index]['bbox'].get_center())
            obj2_center = np.asarray(
                self.objects[obj2_index]['bbox'].get_center())
            return [obj1_center, obj2_center]
        return None

    def __str__(self):
        return '\n'.join([str(edge) for edge in self.edges_by_index.values()])

    def __repr__(self):
        return self.__str__()

    def to_serializable(self):
        s_edges = []
        for (obj1_index, obj2_index), edge in self.edges_by_index.items():
            s_edges.append({
                'obj1_index': obj1_index,
                'obj2_index': obj2_index,
                'rel_type': edge.rel_type,
                'num_detections': edge.num_detections
            })

        # Serialize the object list using its existing method
        s_objects = self.objects.to_serializable()

        return {'edges': s_edges, 'objects': s_objects}

    def load_serializable(self, s_data):
        assert len(self.edges_by_index) == 0 and len(
            self.objects) == 0, 'MapEdgeMapping should be empty when loading'

        # Deserialize the objects list first
        self.objects.load_serializable(s_data['objects'])

        # Rebuild the edges
        for s_edge in s_data['edges']:
            obj1_index = s_edge['obj1_index']
            obj2_index = s_edge['obj2_index']
            rel_type = s_edge['rel_type']
            num_detections = s_edge['num_detections']

            # Create a new edge
            edge = MapEdge(obj1_index, obj2_index, rel_type, num_detections)
            self.edges_by_index[(obj1_index, obj2_index)] = edge

            # Assuming 'id' attribute exists in the objects for UUID key generation
            obj1_uuid = self.objects[obj1_index]['id']
            obj2_uuid = self.objects[obj2_index]['id']
            self.edges_by_uuid[(obj1_uuid, obj2_uuid)] = edge
