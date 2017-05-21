#include <stdio.h>
#include "kernel1.h"


extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
   	extern __shared__ float s_data[];
	
	unsigned int x_global 	= blockIdx.x * blockDim.x + threadIdx.x + 1;
	unsigned int y_global 	= blockIdx.y * blockDim.y + threadIdx.y + 1;
	unsigned int blockWidth = blockDim.x;

	// if out of range, return
  	if( x_global >= width - 1|| y_global >= width - 1 || x_global < 1 || y_global < 1 ) 
		return;

	// set the shared data values in this area;
	
	// if left edge thread
	if(threadIdx.x == 0){
		s_data[0]	 	 = g_dataA[(y_global-1) * floatpitch + x_global - 1]; 			// NW
		s_data[blockWidth+2] 	 = g_dataA[y_global * floatpitch + x_global-1]; 			// W
		s_data[(blockWidth+2)*2] = g_dataA[(y_global+1) * floatpitch + x_global-1];			// SW
	}
	// if right edge thread
	if(threadIdx.x == blockDim.x-1 || x_global == width-2){
		s_data[threadIdx.x+2] 			 = g_dataA[(y_global-1) * floatpitch + x_global + 1];	// NE
		s_data[blockWidth+2+threadIdx.x+2] 	 = g_dataA[y_global * floatpitch + x_global+1]; 	// E
		s_data[(blockWidth+2)*2 + threadIdx.x+2] = g_dataA[(y_global+1) * floatpitch + x_global+1]; 	// SE
	}
	
	// all threads
	s_data[blockWidth+2 + threadIdx.x+1] 	 = g_dataA[y_global * floatpitch + x_global];			// CENTER
	s_data[threadIdx.x+1] 			 = g_dataA[(y_global-1) * floatpitch + x_global];		// N
	s_data[(blockWidth+2)*2 + threadIdx.x+1] = g_dataA[(y_global+1) * floatpitch + x_global];		// S
 				
	// wait for all threads to add values to shared array
	__syncthreads();

	// set the results
	g_dataB[y_global * floatpitch + x_global] = 	(
					        0.1f * s_data[threadIdx.x] +				// NW 
						0.1f * s_data[threadIdx.x+1] +				// N 
						0.1f * s_data[threadIdx.x+2] + 				// NE
						0.1f * s_data[(blockWidth+2) + threadIdx.x] +		// W
						0.2f * s_data[blockWidth+2 + threadIdx.x+1] +	 	// thisBlock
						0.1f * s_data[(blockWidth+2) + threadIdx.x+2] +		// E
						0.1f * s_data[(blockWidth+2)*2 + threadIdx.x] +		// SW
						0.1f * s_data[(blockWidth+2)*2 + threadIdx.x+1] +	// S
						0.1f * s_data[(blockWidth+2)*2 + threadIdx.x+2] 	// SE
					) * 0.95f;  	
}

