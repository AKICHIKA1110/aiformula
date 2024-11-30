#!/usr/bin/env python
import rclpy
import math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from can_msgs.msg import Frame
from std_msgs.msg import Float32
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

        buffer_size = 10
        self.twist_sub = self.create_subscription(
            Twist, 'sub_speed_command', self.twist_callback, buffer_size)
        self.can_pub = self.create_publisher(Frame, 'pub_can', buffer_size)
        self.can_sub = self.create_subscription(
            Frame, 'sub_can', self.can_frame_callback, buffer_size)  # CANフレームの購読
        self.current_pub_left = self.create_publisher(
            Float32, 'motor_current_left', buffer_size)
        self.current_pub_right = self.create_publisher(
            Float32, 'motor_current_right', buffer_size)
        self.voltage_pub = self.create_publisher(
            Float32, 'motor_voltage', buffer_size)
        self.publish_timer = self.create_timer(
            publish_timer_loop_duration, self.publish_canframe_callback)
        self.frame_msg = Frame()

    def twist_callback(self, msg):
        rpm = self.toRefRPM(msg.linear.x, msg.angular.z)
        cmd_right = self.toCanCmd(rpm[0])
        cmd_left = self.toCanCmd(rpm[1])
        can_data = cmd_right + cmd_left
        self.frame_msg.header.frame_id = "can0"        # Default can0
        self.frame_msg.id = 0x210                      # MotorController CAN ID : 0x210
        self.frame_msg.dlc = 8                         # Data length Byte
        self.frame_msg.is_extended = False             # 標準IDを使用
        self.frame_msg.data = can_data

    def publish_canframe_callback(self):
        self.can_pub.publish(self.frame_msg)

    def can_frame_callback(self, msg: Frame):
        # モータ電流と電圧のデータを含むCAN IDをフィルタリング
        if msg.id == 0x111:  # CAN ID (V)
            # データから左右モータの電流を抽出
            current_left = self.extract_current(msg.data[0:4])
            current_right = self.extract_current(msg.data[4:8])

            # 電流データをパブリッシュ
            self.current_pub_left.publish(Float32(data=current_left))
            self.current_pub_right.publish(Float32(data=current_right))

        elif msg.id == 0x111:  # CAN ID (I)
            # データから電圧を抽出
            voltage = self.extract_voltage(msg.data[0:4])

            # 電圧データをパブリッシュ
            self.voltage_pub.publish(Float32(data=voltage))

    def extract_current(self, data_bytes: bytes) -> float:
        # バイトデータを整数に変換し、電流値を計算
        k = 1 #scale setting
        current_raw = int.from_bytes(data_bytes, byteorder='little', signed=True)
        current = current_raw * k  #scale 
        return current

    def extract_voltage(self, data_bytes: bytes) -> float:
        # バイトデータを整数に変換し、電圧値を計算
        voltage_raw = int.from_bytes(data_bytes, byteorder='little', signed=False)
        voltage = voltage_raw * 0.1  # 仮のスケーリング（実際のスケールに合わせて調整）
        return voltage

    def toRefRPM(self, linear_velocity, angular_velocity):  # Calc Motor ref rad/s
        wheel_angular_velocities = np.array([
            (linear_velocity / (self.diameter * 0.5)) + (self.tread / self.diameter) * angular_velocity,  # right[rad/s]
            (linear_velocity / (self.diameter * 0.5)) - (self.tread / self.diameter) * angular_velocity   # left[rad/s]
        ])
        minute_to_second = 60
        rpm = wheel_angular_velocities * (minute_to_second / (2 * math.pi))
        return (rpm * self.gear_ratio).tolist()

    @staticmethod
    def toCanCmd(rpm: float) -> List[int]:
        rounded = round(rpm)
        bytes_data = rounded.to_bytes(4, "little", signed=True)
        return list(bytes_data)

def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    rclpy.spin(motor_controller)
    motor_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
