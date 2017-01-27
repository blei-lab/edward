from setuptools import setup

# import `__version__` from code base
exec(open('edward/version.py').read())

setup(
    name='edward',
    version=__version__,
    description='A library for probabilistic modeling, inference, and '
                'criticism',
    author='Dustin Tran',
    author_email="dustin@cs.columbia.edu",
    packages=['edward', 'edward.criticisms', 'edward.inferences',
              'edward.models', 'edward.stats', 'edward.util'],
    install_requires=['tensorflow>=0.11.0rc0,!=0.12.0rc0,!=0.12.0rc1,!=0.12.0' +
                      ',!=0.12.1',
                      'numpy>=1.7',
                      'six>=1.10.0'],
    extras_require={'stan': ['pystan>=2.0.1.3'],
                    'pymc3': ['pymc3>=3.0'],
                    'model wrappers': ['scipy>=0.16'],
                    'neural networks': ['keras>=1.0.0', 'prettytensor>=0.5.3'],
                    'visualization': ['progressbar>=2.0']},
    url='http://edwardlib.org',
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4'],
)
