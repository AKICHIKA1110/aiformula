import rclpy
import numpy as np
import cv2
import os
from collections import deque
from datetime import datetime
from rclpy.node import Node
from sensor_msgs.msg import Image
from can_msgs.msg import Frame
from cv_bridge import CvBridge
from typing import List

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        self.declare_parameter('wheel.tread')
        self.declare_parameter('wheel.diameter')
        self.declare_parameter('wheel.gear_ratio')
        self.declare_parameter('publish_timer_loop_duration')
        self.tread = self.get_parameter('wheel.tread').get_parameter_value().double_value
        self.diameter = self.get_parameter('wheel.diameter').get_parameter_value().double_value
        self.gear_ratio = self.get_parameter('wheel.gear_ratio').get_parameter_value().double_value
        publish_timer_loop_duration = self.get_parameter(
            'publish_timer_loop_duration').get_parameter_value().double_value
            
        # 日付と時刻をフォーマットしてディレクトリ名を設定
        datetime_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        
        # 保存先のディレクトリの設定
        base_dir_name = f'~/Desktop/rog_data/{datetime_str}'
        self.base_image_dir = os.path.expanduser(base_dir_name)

        # 画像用フォルダを作成
        self.left_image_dir = os.path.join(self.base_image_dir, 'images', 'left_images')
        os.makedirs(self.left_image_dir, exist_ok=True)
        self.bird_image_dir = os.path.join(self.base_image_dir, 'images', 'bird_images')
        os.makedirs(self.bird_image_dir, exist_ok=True)
        self.br = CvBridge()
        
        '''----------------変数宣言----------------'''
        self.sum_loss_center_line = deque(maxlen=3)
        self.left_image_count = 0  # 画像カウンターの初期化
        self.bird_image_count = 0  # 俯瞰画像カウンターの初期化
        self.loss_center_line = 0 # 差分
        
        self.DEFALT_THRESHOLD = 175  # 画像処理関係
        self.DEFALT_IMAGE_BRIGHT = 135
        self.sum_brightness = deque(maxlen=120)
        self.X1, self.Y1 = 270, 200   # コンクリートの明るさチェック
        self.X2, self.Y2 = 320, 440
        self.height = 0
        self.width = 0
        
        self.START_X1 = 0    # 左側最大を探す範囲
        self.END_X1 = 100
        self.START_X2 = 300  # 右側最大を探す範囲
        self.END_X2 = 400
        self.LEFT_DEFAULT_POSITION=80
        self.RIGHT_DEFAULT_POSITION = 320
        self.CAMERA_DIFFERENCE = 3   # カメラが左にあるので中心がずれる分の補正
        self.differ_center_position = 0
        
        buffer_size = 10
        self.left_image_sub = self.create_subscription(Image, '/aiformula_sensing/zed_node/left_image/undistorted', self.left_image_callback, buffer_size)
        self.can_pub = self.create_publisher(Frame, 'pub_can', buffer_size)
        self.publish_timer = self.create_timer(publish_timer_loop_duration, self.publish_canframe_callback)
        self.frame_msg = Frame()
        
    def left_image_callback(self, msg):
        self.cv_left_image = self.br.imgmsg_to_cv2(msg, "bgr8")
        self.left_image_path = os.path.join(self.left_image_dir, f'left_{self.left_image_count:05d}.jpg')
        cv2.imwrite(self.left_image_path, self.cv_left_image)
        self.left_image_count += 1
        rpm = self.ff_controller()
        cmd_right = self.toCanCmd(rpm[0])
        cmd_left = self.toCanCmd(rpm[1])
        can_data = cmd_right + cmd_left
        self.frame_msg.header.frame_id = "can0"
        self.frame_msg.id = 0x210
        self.frame_msg.dlc = 8
        self.frame_msg.data = can_data

    def publish_canframe_callback(self):
        self.can_pub.publish(self.frame_msg)
        
    def ff_controller(self):        
        self.gray_bird_image = self.convert_gray_bird_image()
        
        # 閾値設定
        self.threshold = self.DEFALT_THRESHOLD
        self.brightness_average = self.image_brightness_average()
        self.differ_center_position = self.calculate_difference()
        rpm = self.feedforward()
        
        return rpm
    
    def convert_gray_bird_image(self):
        converted_gray_image = cv2.cvtColor(self.cv_left_image, cv2.COLOR_BGR2GRAY)
        cropped_frame = converted_gray_image[self.Y1:self.Y2, self.X1:self.X2]
        brightness = np.mean(cropped_frame)
        self.sum_brightness.append(brightness)
        self.ave_brightness = sum(self.sum_brightness) / len(self.sum_brightness)

        points_original = np.float32([[250, 200], [390, 200], [0, 250], [640, 250]])
        points_transformed = np.float32([[0, 0], [400, 0], [0, 300], [400, 300]])
        matrix = cv2.getPerspectiveTransform(points_original, points_transformed)
        bird_eye = cv2.warpPerspective(converted_gray_image, matrix, (400, 300))
        self.height, self.width = bird_eye.shape[:2]
        
        self.bird_image_path = os.path.join(self.bird_image_dir, f'left_{self.bird_image_count:05d}.jpg')
        cv2.imwrite(self.bird_image_path, bird_eye)
        self.bird_image_count += 1
        
        return bird_eye
        
    def image_brightness_average(self):
        sum_ave_column = []
        bottom_quarter = self.gray_bird_image[6 * self.height // 8 :, :]
        self.x_positions = range(self.width)
        for x in self.x_positions:
            colmun = bottom_quarter[:, x]
            ave_column = sum(colmun) / self.height
            sum_ave_column.append(ave_column)
            
        return sum_ave_column
            
    def calculate_difference(self):
        left_subset = self.brightness_average[self.START_X1:self.END_X1]
        left_max_value = max(left_subset)
        left_max_index = self.brightness_average.index(left_max_value) + 1
        right_subset = self.brightness_average[self.START_X2:self.END_X2] 
        right_max_value = max(right_subset)
        right_max_index = self.brightness_average.index(right_max_value) + 1
        
        return (left_max_index + right_max_index) / 2 - 200 + self.CAMERA_DIFFERENCE  
    
    def feedforward(self):
        left_rpm, right_rpm = 90, 90
        GAIN = 1.5
        self.sum_loss_center_line.append(self.differ_center_position)
        ave_sum_loss_center_line = sum(self.sum_loss_center_line) / len(self.sum_loss_center_line)
       
        if ave_sum_loss_center_line > 0:
            right_rpm += abs(ave_sum_loss_center_line) * GAIN
        else:
            left_rpm += abs(ave_sum_loss_center_line) * GAIN

        rpm_list = [right_rpm, left_rpm]
        
        return rpm_list
    
    @staticmethod
    def toCanCmd(rpm: float) -> List[int]:
        rounded = round(rpm)
        bytes = rounded.to_bytes(4, "little", signed=True)
        return list(bytes)
        
def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    rclpy.spin(motor_controller)
    motor_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
