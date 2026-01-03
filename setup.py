from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros2_network_predictor'

setup(
    name=package_name,
    version='0.0.0',
    # Automatically find the main package and sub-packages (models, forecasters)
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        # Ensure the .pth model weights are installed
        (os.path.join('lib', package_name, 'models'), glob(os.path.join(package_name, 'models', '*.pth'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leticia',
    maintainer_email='leticia@todo.com',
    description='AI-based latency forecasting for ROS 2 using TCN models',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # The Manager Node
            'network_predictor_node = ros2_network_predictor.network_predictor_node:main',
        ],
    },
    # Hardcode the pyenv interpreter for the entry point scripts
    options={
        'build_scripts': {
            'executable': '/home/leticia/.pyenv/versions/ros2-predictor/bin/python3',
        },
    },
)