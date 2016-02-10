try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'blackbox',
    'description': 'Black box (approximate) inference for probabilistic models',
    'author': 'Dustin Tran',
    'author_email': 'dustin@cs.columbia.edu',
    'version': '0.1',
    'packages': ['blackbox'],
    'scripts': [],
}

setup(**config)
