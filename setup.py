import os
from setuptools import setup, find_packages

os.chdir('kg_som')

setup(name='kg_som',
      version='0.1',
      url='https://bitbucket.org/kireygroup/kg_ml_som',
      license='MIT',
      author='Riccardo Sayn',
      author_email='riccardo.sayn@kireygroup.com',
      description='A PyTorch+Fastai based implementation of Self-Organizing Maps',
      packages=find_packages(),
      install_requires=['fastai', 'sklearn', 'kmeans_pytorch', 'seaborn', 'smart-open==1.8.0', 'gensim==3.7.1'],
      long_description=open('README.md').read(),
      zip_safe=False)
