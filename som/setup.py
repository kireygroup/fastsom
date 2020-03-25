from setuptools import setup, find_packages

setup(name='.som',
      version='0.1',
      url='https://bitbucket.org/kireygroup/kg_ml_som/kg_ml_som/som',
      license='MIT',
      author='Riccardo Sayn',
      author_email='riccardo.sayn@kireygroup.com',
      description='Add static script_dir() method to Path',
      packages=find_packages('som'),
      package_dir={'': 'som'},
      long_description=open('README.md').read(),
      zip_safe=False)
