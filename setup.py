from setuptools import setup

setup(
        name='sic',
        version='0.1',
        descriptio='tools for private ensemble learning.',
        url='',
        author='Brian Goodchild',
        author_email='bgoodc@cs.columbia.edu',
        liscense='',
        packages=['sic', 'sic.testing'],
        install_requires=[
            'numpy',
            'pandas',
            'tensorflow',
            'keras',
            'scikit-learn'
        ],
        dependency_links=['https://github.com/tensorflow/privacy.git']
)

