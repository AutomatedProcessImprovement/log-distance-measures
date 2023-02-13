from setuptools import setup

setup(
    name='log_similarity_metrics',
    version='0.8.1',
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'pandas',
        'scipy',
        'numpy',
        'jellyfish',
        'dtw'
    ]
)
