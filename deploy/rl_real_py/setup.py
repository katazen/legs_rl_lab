from setuptools import find_packages, setup
from glob import glob

package_name = 'rl_real_py'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/configs', glob('configs/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='woan',
    maintainer_email='woan@todo.todo',
    description='Config-driven RL policy real-robot deployment node (rl_real_common).',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            "rl_real_common = rl_real_py.rl_real_common:main",
        ],
    },
)
