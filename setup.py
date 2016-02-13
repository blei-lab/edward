from setuptools import setup

setup(
    name='blackbox',
    version='0.1',
    description='Black box inference for probabilistic models',
    author='Dustin Tran',
    author_email="dustin@cs.columbia.edu",
    packages=['blackbox', 'blackbox.stats'],
    install_requires=['tensorflow>=0.6.0', 'numpy>=1.7', 'scipy>=0.16'],
    url='https://github.com/Blei-Lab/blackbox',
    license='MIT',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7'],
)
