import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, sources,includes):

    return CUDAExtension(
        name='{}'.format(name),
        sources=[p for p in sources],
        include_dirs=[i for i in includes],
        extra_compile_args={
            'cxx':  [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]})
#-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D_GLIBCXX_USE_CXX11_ABI=1
sources=[]
sources.extend(glob.glob('src/*.cu'))
sources.extend(glob.glob('src/*.cpp'))

with open("README.md", "r",encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='DSConv_CUDA',
    version='1.0.0',
    author='QibiaoWu',
    author_email='QibiaoWu1116@163.com',
    url='https://www.github.com',
    description="cuda implementation of DSConv",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[
        make_cuda_ext(name='DSCONV_CUDA',
                      sources=sources,
                      includes=['src'])
    ],
    py_modules=['DSConv_CUDA'],
    classifiers=(
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Operating System :: POSIX :: Linux',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',
    ),
    install_requires=['torch>=1.3'],
    keywords=["pytorch", "cuda", "deform"],
    cmdclass={'build_ext': BuildExtension}, zip_safe=False)
