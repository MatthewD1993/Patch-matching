from distutils.core import setup, Extension
import numpy.distutils.misc_util
extension_mod = Extension("patchselect",
	include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()+['/opt/Python-3.6.1/include/''/tmpbig/isg/opencv/modules/imgproc/include/', './'],
	libraries = ['opencv_core', 'opencv_imgproc','opencv_highgui',],
	library_dirs = ['/usr/local/lib'],
	sources =["main.cpp",  "basicdefinitions.cpp",],
	extra_compile_args=['-std=c++11'])
setup(name="patchselect", ext_modules=[extension_mod])