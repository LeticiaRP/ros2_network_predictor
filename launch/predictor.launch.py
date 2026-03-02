from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():


    # Define command-line arguments
    scenario_arg = DeclareLaunchArgument('scenario', default_value='h2h')
    freq_arg = DeclareLaunchArgument('freq', default_value='20.0')
    qos_arg = DeclareLaunchArgument('reliable', default_value='True')
    safety_arg = DeclareLaunchArgument('safety', default_value='0.7')
    failure_arg = DeclareLaunchArgument('failure', default_value='2.0')



    # Setup the Node
    predictor_node = Node(
        package='ros2_network_predictor',
        executable='network_predictor_node',
        name='network_predictor_node',
        output='screen',
        parameters=[{
            'scenario': LaunchConfiguration('scenario'),
            'frequency_hz': LaunchConfiguration('freq'),
            'qos_reliable': LaunchConfiguration('reliable'),
            'safety_threshold': LaunchConfiguration('safety'),
            'failure_threshold': LaunchConfiguration('failure'),
        }]
    )



    return LaunchDescription([
        scenario_arg,
        freq_arg,
        qos_arg,
        safety_arg,
        failure_arg,
        predictor_node
    ])