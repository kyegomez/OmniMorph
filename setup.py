from setuptools import setup, find_packages

setup(
  name = 'omnimorph',
  packages = find_packages(exclude=[]),
  version = '0.0.7',
  license='MIT',
  description = 'OmniMorph - Pytorch',
  author = 'Agora',
  author_email = 'kye@apac.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kyegomez/OmniMorph',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)