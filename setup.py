from setuptools import setup

setup(
    name='log_distance_measures',
    version='0.12.0',
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'pandas',
        'scipy',
        'numpy',
        'jellyfish',
        'pulp'
    ]
)
