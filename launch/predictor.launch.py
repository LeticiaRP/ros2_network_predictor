from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    
    topic_arg = DeclareLaunchArgument('topic', default_value='/benchmark/latency/h2h')
    pkg_arg = DeclareLaunchArgument('pkg', default_value='std_msgs')
    type_arg = DeclareLaunchArgument('type', default_value='Float32')
    freq_arg = DeclareLaunchArgument('freq', default_value='20.0')
    cal_arg = DeclareLaunchArgument('cal', default_value='100')

    predictor_node = Node(
        package='ros2_network_predictor',
        executable='network_predictor_node',
        output='screen',
        parameters=[{
            'topic': LaunchConfiguration('topic'),
            'msg_pkg': LaunchConfiguration('pkg'),
            'msg_type': LaunchConfiguration('type'),
            'frequency_hz': LaunchConfiguration('freq'),
            'calibration_steps': LaunchConfiguration('cal'),
        }]
    )

    return LaunchDescription([topic_arg, pkg_arg, type_arg, freq_arg, cal_arg, predictor_node])