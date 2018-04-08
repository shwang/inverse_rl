from setuptools import setup, find_packages
import sys, os.path

# Don't import module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'airl'))
from version import VERSION

setup(
    name='adversarial-irl',
    version=VERSION,
    packages=find_packages(exclude=['scripts', 'tabular_maxent_irl']),
    package_data={
        'envs': ['*.xml'],
    },
    install_requires=[
        'rllab>=0.1.0', 'gym>=0.10.4', 'numpy>=1.12', 'tensorflow>=1.4.0',
    ],

    # Metadata
    author='Justin Fu, Adam Gleave',
    license='GPL 3.0',
    description='Implementation of adversarial IRL (Fu et al, 2017)',
    url='https://github.com/AdamGleave/inverse_rl',
)