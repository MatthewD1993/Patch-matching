#include "Python.h"
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "patchselect.h"
#include <opencv2/core/core.hpp>
#include <ctime>
using namespace cv;


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
	clock_t start_init = clock();
	_ps = new patchselect(image1, image2, flow, cntImages, patchsize, scale, offset);
	cout << "Time to initialize: (s):"<< ((float)(clock() - start_init))/CLOCKS_PER_SEC  << endl;
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

//	clock_t startTime = clock();
	v->add(cnt);
//	clock_t time_chunk_over = clock();
//	cout << "Time to create patches chunk (s): " << ((float)(time_chunk_over - startTime))/CLOCKS_PER_SEC <<endl;

	// Create a data block. Shape(num_pairs, 2, height, width, num_channels)
	int nd=6;
	npy_intp dims[]={cnt, 2, 2, v->_patchsize, v->_patchsize, v->channels};

//    cout<<"Simple new!"<<endl;

	out_array = PyArray_SimpleNew(nd, dims, NPY_FLOAT32);

	a = (float *)PyArray_DATA(out_array);
//	cout << "Create ptr->......"<<endl;


//	clock_t time_start_mem = clock();
	v->createPyArrayPtr(a);
//	cout << "Time to create memory: (s):"<< ((float)(clock()-time_start_mem))/CLOCKS_PER_SEC  << endl;

//	cout << "Data Loaded!" << endl;
	return out_array;
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
