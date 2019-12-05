/* This is the header file for some cuda helper 
 *
 * Author: Guodong Chen
 * Email: cgderic@umich.edu
 * Date:  12/04/2019
 */

#ifndef _DG_CUDA_HELPER_
#define _DG_CUDA_HELPER_

#include "cuda_runtime.h"
#include "stdio.h"
/* Error checking macro */
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  printf("Cuda Error: %s\n", cudaGetErrorString(x)); \
  exit(EXIT_FAILURE);}} while(0)

#endif
