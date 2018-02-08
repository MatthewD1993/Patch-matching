#include "Python.h"
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "patchselect.h"
#include <opencv2/core/core.hpp>
using namespace cv;
// static void free_patchselect(void* p)
// {
// 	printf("Free the patchselect.\n");
// 	patchselect * v = (patchselect *) p;
// 	free(v);
// }


//static inline PyObject* pyObjectFromRefcount(const int* refcount)
//{
//    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
//}
//
//static inline int* refcountFromPyObject(const PyObject* obj)
//{
//    return (int*)((size_t)obj + REFCOUNT_OFFSET);
//}
//
//
//class NumpyAllocator : public MatAllocator
//{
//public:
//    NumpyAllocator() {}
//    ~NumpyAllocator() {}
//
//    void allocate(int dims, const int* sizes, int type, int*& refcount,
//                  uchar*& datastart, uchar*& data, size_t* step)
//    {
//        PyEnsureGIL gil;
//
//        int depth = CV_MAT_DEPTH(type);
//        int cn = CV_MAT_CN(type);
//        const int f = (int)(sizeof(size_t)/8);
//        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
//                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
//                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
//                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
//        int i;
//        npy_intp _sizes[CV_MAX_DIM+1];
//        for( i = 0; i < dims; i++ )
//        {
//            _sizes[i] = sizes[i];
//        }
//
//        if( cn > 1 )
//        {
//            _sizes[dims++] = cn;
//        }
//
//        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
//
//        if(!o)
//        {
//            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
//        }
//        refcount = refcountFromPyObject(o);
//
//        npy_intp* _strides = PyArray_STRIDES(o);
//        for( i = 0; i < dims - (cn > 1); i++ )
//            step[i] = (size_t)_strides[i];
//        datastart = data = (uchar*)PyArray_DATA(o);
//    }
//
//    void deallocate(int* refcount, uchar*, uchar*)
//    {
//        PyEnsureGIL gil;
//        if( !refcount )
//            return;
//        PyObject* o = pyObjectFromRefcount(refcount);
//        Py_INCREF(o);
//        Py_DECREF(o);
//    }
//};
//
//NumpyAllocator g_numpyAllocator;
//
//
//PyObject* toNDArray(const cv::Mat& m)
//{
//    if( !m.data )
//        Py_RETURN_NONE;
//    Mat temp, *p = (Mat*)&m;
//    if(!p->refcount || p->allocator != &g_numpyAllocator)
//    {
//        temp.allocator = &g_numpyAllocator;
//        m.copyTo(temp);
//        p = &temp;
//    }
//    p->addref();
//    return pyObjectFromRefcount(p->refcount);
//}












void Destroyer(PyObject *capsule)
{
    auto rawPtr = static_cast<patchselect*>(PyCapsule_GetPointer(capsule, nullptr));
    delete rawPtr;
}

static PyObject* init(PyObject* self, PyObject* args)
{
	patchselect* _ps;
	char *image1, *image2, *flow;
	int cntImages;
	int patchsize = 32;
	int scale = 1;
	int offset=0;
	// Parser might cause error
	if (!PyArg_ParseTuple(args, "sssiiii", &image1, &image2, &flow, &cntImages, &patchsize, &scale, &offset)){
		return NULL;
	}
	_ps = new patchselect(image1, image2, flow, cntImages, patchsize, scale, offset);
	cout << "Initialization success!" <<endl;
	return PyCapsule_New(_ps, nullptr, Destroyer);
}


static PyObject* requestNewData(PyObject* self, PyObject* args)
{
	patchselect * v;
	PyObject * p;
	PyObject * out_array;
	float *a;
	int cnt;
	if (!PyArg_ParseTuple(args, "Oi", &p, &cnt)){
		return NULL;
	}
	v = static_cast<patchselect*>(PyCapsule_GetPointer(p, nullptr));
	
	v->reset();
	v->add(cnt);
	// Create a data block. Shape(num_pairs, 2, height, width, num_channels)
	int nd=6;
	npy_intp dims[]={cnt, 2, 2, v->channels, v->_patchsize, v->_patchsize};

	out_array = PyArray_SimpleNew(nd, dims, NPY_FLOAT32);

	a = (float *)PyArray_DATA(out_array);
	cout << "Create ptr->......"<<endl;
	v->createPyArrayPtr(a);
	cout << "Data Loaded!" << endl;
	return out_array;


//	v->reset();
//	v->add(cnt);

//	Mat imagexx;
//	cvtColor(v->_pos[0].second[0], imagexx, COLOR_Lab2BGR);
//	cv::imshow("new", imagexx);
//	cv::waitKey(0);


//	// Create a data block. Shape(num_pairs, 2, height, width, num_channels)
//	int nd=3;
//	npy_intp dims[]={ 200,200, 3};
//    cout<<"Simple new!"<<endl;
//
//	out_array = PyArray_SimpleNew(nd, dims, NPY_FLOAT32);
//	cout<<"Be sure"<<endl;
//	a = (float *)PyArray_DATA(out_array);
//	cout << "Create ptr->......"<<endl;
//	v->createPyArrayPtr_test(a);




//    Mat img(200,200,CV_32FC3, (void *)a);
//    Mat temp;
//    cvtColor(img, temp, COLOR_Lab2BGR);
//    cv::imshow("converted", temp);
//    cv::waitKey(5000);


//	cout << "Data Loaded!" << endl;
//	return out_array;





}


static PyMethodDef PatchselectMethods[] = {
  { "init", init, METH_VARARGS, "Initialize patch selector" },
  { "newData", requestNewData, METH_VARARGS, "Return 5D numpy array." },
  { NULL, NULL, 0, NULL }
};

static PyModuleDef patchselectmodule={
	PyModuleDef_HEAD_INIT,
	"patchselect", // name of module
	NULL,
	-1,
	PatchselectMethods
};

PyMODINIT_FUNC
PyInit_patchselect(void)
{
	import_array();
	return PyModule_Create(&patchselectmodule);

}
