from setuptools import setup, find_packages

setup(
    name='vehicle_inference',
    version='0.1.0',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': [
            'vehicle-inference = scripts.run_inference:main',
        ]
    },
    author='Your Name',
    description='Vehicle detection and VLM inference package',
    license='MIT',
)