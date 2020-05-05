from setuptools import setup, find_packages

# Retrieve description from README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(name='fastsom',
      version='0.1.7',
      url='https://github.com/kireygroup/fastsom',
      download_url='https://github.com/kireygroup/fastsom/archive/v0.1.7.tar.gz',
      license='MIT',
      author='Riccardo Sayn',
      author_email='riccardo.sayn@kireygroup.com',
      description='A PyTorch and Fastai based implementation of Self-Organizing Maps',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=['fastai', 'sklearn', 'kmeans_pytorch', 'seaborn', 'smart-open==1.8.0', 'gensim==3.7.1'],
      keywords=['self-organizing-map', 'fastai', 'pytorch', 'python'],
      zip_safe=False,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ])
