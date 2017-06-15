from setuptools import setup

# import ``__version__`` from code base
exec(open('edward/version.py').read())

setup(
    name='edward',
    version=__version__,
    description='A library for probabilistic modeling, inference, and '
                'criticism',
    author='Dustin Tran',
    author_email="dustin@cs.columbia.edu",
    packages=['edward', 'edward.criticisms', 'edward.inferences',
              'edward.models', 'edward.util', 'edward.inferences.conjugacy'],
    install_requires=['numpy>=1.7',
                      'six>=1.10.0'],
    extras_require={
        'tensorflow': ['tensorflow>=1.2.0rc0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.2.0rc0'],
        'neural networks': ['keras>=1.0.0,<=2.0.4', 'prettytensor>=0.7.4'],
        'notebooks': ['jupyter>=1.0.0'],
        'visualization': ['matplotlib>=1.3',
                          'pillow>=3.4.2',
                          'seaborn>=0.3.1']},
    tests_require=['pytest', 'pytest-pep8'],
    url='http://edwardlib.org',
    keywords='machine learning statistics probabilistic programming tensorflow',
    license='Apache License 2.0',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: Apache Software License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4'],
)
