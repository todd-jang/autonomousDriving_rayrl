from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='robot_localization', executable='ekf_node',
             name='ekf_filter_node', output='screen',
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                         'config/ekf.yaml']),
        Node(package='nav2_bringup', executable='bringup_launch.py',
             name='nav2', output='screen',
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
