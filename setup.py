'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games
  
Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------
'''
  
from setuptools import setup, find_packages
  
setup(
    name = 'rufus',
    version = '0.0.1',
    packages = find_packages(where='src'),
    package_dir = {'': 'src'},
    install_requires = [
        'treelib >= 1.5.5',
        'matplotlib >= 2.2.2',
        'numpy >= 1.14.3'
    ],
      
    test_suite='rufus.test',
      
    author = 'Jeffrey Wallace',
    description = 'Modeling and analysis of differential games',
    license = 'MIT'
)
