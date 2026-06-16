from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    engine = LaunchConfiguration('engine')
    conf = LaunchConfiguration('conf')
    min_observations = LaunchConfiguration('min_observations')

    return LaunchDescription([
        DeclareLaunchArgument('engine', default_value='best.engine'),
        DeclareLaunchArgument('conf', default_value='0.45'),
        # Use 1 for immediate pose output per detection; use 4 for more stable confirmed tracks.
        DeclareLaunchArgument('min_observations', default_value='1'),
        Node(
            package='yolo_inference',
            executable='yolo_cam_optimized',
            name='yolo_detector_node',
            output='screen',
            arguments=['--engine', engine, '--conf', conf],
            parameters=[{
                'min_observations': ParameterValue(min_observations, value_type=int),
            }],
        ),
    ])
