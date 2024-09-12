import sqlite3
from tf2_msgs.msg import TFMessage
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from node_single import SingleRobotSLAMNode
import rclpy
import time
import numpy as np
import tf2_geometry_msgs
import geometry_helper
from rclpy.duration import Duration
from tf2_ros import LookupException
from tf2_ros import ExtrapolationException, ConnectivityException
from tf2_ros.buffer import Buffer


def find_first_vl_odom_tf(select_results, topic_id2name):
    buf = Buffer(cache_time=Duration(seconds=10))
    for message in select_results:
        _, topic_id, _, serialized_msg = message
        if topic_id2name[topic_id] != "/tf":
            continue
        msg = deserialize_message(
            serialized_msg,
            TFMessage,
        )
        for transform in msg.transforms:
            buf.set_transform(
                transform,
                "default_authority",
            )
            if transform.header.frame_id != "vl":
                continue
            try:
                vl_to_odom = buf.lookup_transform(
                    "vl",
                    "odom",
                    transform.header.stamp,
                    timeout=Duration(seconds=0.),
                )
            except (LookupException, ExtrapolationException,
                    ConnectivityException):
                continue
            return vl_to_odom.transform
    raise ValueError("Cannot find transform from vl to odom")


def odom_to_first_vl_coord(first_vl_odom_tf, odom):
    """NOTE
    Somehow Around robot publishes transform 'from' baselink 'to' odom.
    It should be inversed...
    """
    quat1 = np.array([
        first_vl_odom_tf.rotation.x,
        first_vl_odom_tf.rotation.y,
        first_vl_odom_tf.rotation.z,
        first_vl_odom_tf.rotation.w,
    ])
    euler1 = geometry_helper.quat2euler(quat1)
    xyth1 = np.array([
        first_vl_odom_tf.translation.x,
        first_vl_odom_tf.translation.y,
        euler1[2],
    ])

    odom_mat = np.eye(4, 4)
    odom_mat[0, 3] = odom.transform.translation.x
    odom_mat[1, 3] = odom.transform.translation.y

    quat2 = np.array([
        odom.transform.rotation.x,
        odom.transform.rotation.y,
        odom.transform.rotation.z,
        odom.transform.rotation.w,
    ])
    odom_rot_mat = geometry_helper.quat2mat(quat2)

    odom_mat = np.matmul(odom_mat, odom_rot_mat)
    inv_odom_mat = np.linalg.inv(odom_mat)
    inv_odom_euler = geometry_helper.mat2euler(inv_odom_mat)
    inv_odom_xy = inv_odom_mat[:3, 3]

    xyth2 = np.array([
        inv_odom_xy[0],
        inv_odom_xy[1],
        inv_odom_euler[2],
    ])

    new_xyth = geometry_helper.transform(xyth2, xyth1, rel_to_abs=True)

    euler = np.array([0, 0, new_xyth[2]])
    quat = geometry_helper.euler2quat(euler)

    odom.transform.rotation.x = quat[0]
    odom.transform.rotation.y = quat[1]
    odom.transform.rotation.z = quat[2]
    odom.transform.rotation.w = quat[3]
    odom.transform.translation.x = new_xyth[0]
    odom.transform.translation.y = new_xyth[1]

    odom.child_frame_id = "base_link"
    odom.header.frame_id = "odom"

    return odom


def main():
    rclpy.init()

    bag_file = "test_data/2f_bag/for_slam_toolbox_0.db3"
    conn = sqlite3.connect(bag_file, check_same_thread=False)

    query = "SELECT * FROM topics;"
    all_topics = conn.execute(query).fetchall()
    # topic_name2id = {topic[1]: topic[0] for topic in all_topics}
    topic_id2name = {topic[0]: topic[1] for topic in all_topics}
    print(topic_id2name)

    query = "SELECT * FROM messages"
    select_results = conn.execute(query).fetchall()

    slam_node = SingleRobotSLAMNode()

    current_stamp = 0.
    last_proc_stamp = 0.

    first_vl_odom_tf = None

    elapsed_times = []

    if slam_node._odom_frame == "odom":
        first_vl_odom_tf = find_first_vl_odom_tf(select_results, topic_id2name)

    for message in select_results:
        _, topic_id, _, serialized_msg = message

        if topic_id2name[topic_id] == "/tf":
            msg = deserialize_message(
                serialized_msg,
                TFMessage,
            )
            if msg.transforms:
                current_stamp = (msg.transforms[0].header.stamp.sec +
                                 msg.transforms[0].header.stamp.nanosec * 1e-9)
            for transform in msg.transforms:
                # print(transform.header.frame_id, transform.child_frame_id)
                if (first_vl_odom_tf and transform.child_frame_id == "odom"
                        and transform.header.frame_id == "base_link"):
                    # print(transform)
                    transform = odom_to_first_vl_coord(first_vl_odom_tf,
                                                       transform)
                slam_node._tf_buffer.set_transform(
                    transform,
                    "default_authority",
                )
        elif topic_id2name[topic_id] == "/tf_static":
            msg = deserialize_message(
                serialized_msg,
                TFMessage,
            )
            if msg.transforms:
                current_stamp = (msg.transforms[0].header.stamp.sec +
                                 msg.transforms[0].header.stamp.nanosec * 1e-9)
            for transform in msg.transforms:
                slam_node._tf_buffer.set_transform_static(
                    transform,
                    "default_authority",
                )
        elif topic_id2name[topic_id] == "/fused/lscan":
            msg = deserialize_message(
                serialized_msg,
                LaserScan,
            )
            msg.range_min = 0.3
            msg.range_max = 5.0
            current_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            slam_node._laser_callback(msg)

        if current_stamp - last_proc_stamp >= 0.05:
            result, elapsed = slam_node.run_once()
            if result:
                print("elapsed", elapsed)
                elapsed_times.append(elapsed)
            last_proc_stamp = current_stamp

    print("Statistics")
    elapsed_times = np.array(elapsed_times[1:])
    elapsed_percentiles = np.percentile(
        elapsed_times, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print("elapsed mean", elapsed_times.mean())
    print("elapsed std", elapsed_times.std())
    print("elapsed percentiles", elapsed_percentiles)


if __name__ == "__main__":
    main()
