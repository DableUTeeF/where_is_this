#include <cstdlib>
#include "hamming_distance.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

int hamming_distance( int n, unsigned char *x, unsigned char *y ) 
{
	int c = 0;
	for( int i = 0; i < n; i++ ) 
	{
		c += HAMMING_DISTANCE[x[i]][y[i]];
	}
	return c;
}

PYBIND11_MODULE(hamming_distance, m) {
    m.def("hamming_distance", &hamming_distance, "A function that calculate hamming_distance");
}
