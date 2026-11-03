from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(
        package='wit_ros2_imu', executable='wit_ros2_imu', name='imu',
        parameters=[{'port': '/dev/tty_imu', 'baud': 115200}], output='screen',
    )])
