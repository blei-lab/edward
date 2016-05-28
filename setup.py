from setuptools import setup

setup(
    name='edward',
    version='1.0.5',
    description='A library for probabilistic modeling, inference, and criticism',
    author='Dustin Tran',
    author_email="dustin@cs.columbia.edu",
    packages=['edward', 'edward.stats'],
    install_requires=['tensorflow>=0.7.0', 'numpy>=1.7', 'scipy>=0.16'],
    extras_require = {'stan': ['pystan>=2.0.1.3'],
                      'pymc3': ['pymc3>=3.0'],
                      'neural networks': ['keras>=1.0.0', 'prettytensor>=0.5.3'],
                      'visualization': ['progressbar>=2.0']},
    url='https://github.com/blei-lab/edward',
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                 'Programming Language :: Python :: 2.7'],
)
