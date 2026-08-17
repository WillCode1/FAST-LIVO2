import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('fast_livo')

    default_param_config = os.path.join(pkg_dir, 'config', 'ros2_param.yaml')
    default_rviz_config = os.path.join(pkg_dir, 'rviz_cfg', 'mapping_ros2.rviz')
    default_camera_config = os.path.join(pkg_dir, 'config', 'camera_pinhole.yaml')

    declare_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='是否启动 RViz2'
    )
    declare_camera_config = DeclareLaunchArgument(
        'camera_config',
        default_value=default_camera_config,
        description='相机配置文件路径（用于 -camera_config 参数）'
    )

    rviz_enabled = LaunchConfiguration('rviz')
    camera_config = LaunchConfiguration('camera_config')

    fast_livo_node = Node(
        package='fast_livo',
        executable='fastlivo_mapping',
        prefix=['stdbuf -o L'],
        output='screen',
        parameters=[default_param_config],
        arguments=['-camera_config', camera_config]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', default_rviz_config],
        condition=IfCondition(rviz_enabled)
    )

    return LaunchDescription([
        declare_rviz,
        declare_camera_config,
        fast_livo_node,
        rviz_node
    ])