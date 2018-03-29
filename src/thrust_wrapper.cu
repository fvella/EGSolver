#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
/*
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/version.h>
#include <vector>
*/
#include <cuda.h>
#include "thrust_wrapper.h"

//#include <stdio.h>

extern "C" void prefix_summa(int *dev_data, int len)
{
	//std::cout << "This is C++ code!" << std::endl;
	thrust::device_ptr<int> dptr(dev_data);
	thrust::inclusive_scan(dptr, dptr + len, dptr);
}

extern "C" void remove_nulls(int *dev_data, int len, int *prefixLen)
{

	thrust::device_ptr<int> dptr(dev_data);
	//thrust::raw_pointer_cast( );
	//thrust::remove(dptr, dptr + len, 0);
	int * raw_ptr = thrust::raw_pointer_cast(thrust::remove(dptr, dptr + len, 0));
	*prefixLen=(int)((raw_ptr)-dev_data);
	//printf(" XXX1 : %d\n",(int)((raw_ptr)-dev_data));


}
