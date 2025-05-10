import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
            
        for ext in self.extensions:
            self.build_extension(ext)
            
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
            
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DBUILD_JAVA_BINDINGS=OFF'  # Only build Python for pip install
        ]
        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
            
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        # Run cmake on the parent directory (project root)
        subprocess.check_call(['cmake', os.path.join(self.sourcedir, '..')] + cmake_args, 
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, 
                              cwd=self.build_temp)

setup(
    name='pycudbscan',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Python bindings for GPU-accelerated DBSCAN clustering using CUDA',
    long_description=open(os.path.join(os.path.dirname(__file__), '..', 'README.md')).read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/cudbscan',
    packages=find_packages(),
    ext_modules=[CMakeExtension('pycudbscan_core')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering',
    ],
)