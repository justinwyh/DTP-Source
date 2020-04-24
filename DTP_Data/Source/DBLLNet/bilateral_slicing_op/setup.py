from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bilateral_slicing',
    ext_modules=[
        CUDAExtension('bilateral_slicing', [
            'bilateral_slicing.cpp',
            'bilteral_slicing_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
