import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from realsense2_camera_msgs.msg import RGBD
from cv_bridge import CvBridge
from typing import Optional


class RGBDSubscriber(Node):
    def __init__(self) -> None:
        super().__init__('rgbd_subscriber')

        # ROS2 Subscription to the RGBD topic
        self.subscription = self.create_subscription(
            RGBD,
            '/robot0/realsense0/rgbd',  # 토픽 이름을 launch 파일에 맞게 수정
            self.rgbd_callback,
            10)

        # CvBridge for converting ROS images to OpenCV/numpy format
        self.bridge = CvBridge()

        # To store the rgb and depth images
        self.rgb_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None

    def rgbd_callback(self, msg: RGBD) -> None:
        # Extract RGB image from RGBD message
        self.rgb_image = self.convert_image_to_np(msg.rgb, "rgb8")
        self.get_logger().info('RGB Image received.')

        # Extract Depth image from RGBD message
        self.depth_image = self.convert_image_to_np(msg.depth, "16UC1")  # Assuming 16-bit depth
        self.get_logger().info('Depth Image received.')

    def convert_image_to_np(self, img_msg: Image, encoding: str) -> np.ndarray:
        """Convert ROS Image message to numpy array using CvBridge."""
        try:
            # Use cv_bridge to convert to numpy array
            np_image = self.bridge.imgmsg_to_cv2(img_msg, encoding)
            return np_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return np.array([])  # Return empty array on failure


def main(args=None) -> None:
    rclpy.init(args=args)

    # Initialize node and start spinning
    rgbd_subscriber = RGBDSubscriber()
    rclpy.spin(rgbd_subscriber)

    # Shutdown on exit
    rgbd_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
