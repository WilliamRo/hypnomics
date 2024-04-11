from setuptools import setup, find_packages


# Specify version
VERSION = '1.0.0.dev1'


# Preprocess
print('Running setup.py for hypnomics-v' + VERSION + ' ...')
print('-' * 79)


# Run setup
def readme():
  with open('README.md', 'r') as f:
    return f.read()

# Submodules will be included as package data, specified in MANIFEST.in
setup(
  name='hypnomics',
  packages=find_packages(),
  include_package_data=True,
  version=VERSION,
  description='A package for sleep data analysis.',
  long_description=readme(),
  long_description_content_type='text/markdown',
  author='Wei Luo',
  author_email='luo.wei@zju.edu.cn',
  url='https://github.com/WilliamRo/hypnomics',
  download_url='https://github.com/WilliamRo/hypnomics/tarball/v' + VERSION,
  license='Apache-2.0',
  keywords=['sleep', 'EEG', 'hypnogram', 'sleep stage', 'sleep data analysis'],
  classifiers=[
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Utilities",
  ],
)
