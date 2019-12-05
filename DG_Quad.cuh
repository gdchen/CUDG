/**
 * Dunavant points generated with .m code written by John Burkard

 * http://people.scs.fsu.edu/~burkardt/f_src/dunavant/dunavant.html
 * 
 *1. David Dunavant,
 *   High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle,
 *   International Journal for Numerical Methods in Engineering,
 *   Volume 21, 1985, pages 1129-1148.
 *2. James Lyness, Dennis Jespersen,
 *   Moderate Degree Symmetric Quadrature Rules for the Triangle,
 *   Journal of the Institute of Mathematics and its Applications,
 *   Volume 15, Number 1, February 1975, pages 19-32.

 *Note, the coordinates in the x[] vectors are stored unrolled in
 *sequential x,y pairs, e.g. [x1 y1  x2 y2  x3 y3 ... ]

 */

/* The file has been modified to fit the use here, by Guodong Chen
 * Email: cgderic@umich.edu
 * Last modified: 12/05/2019
 */

#ifndef _DG_Quad_
#define _DG_Quad_



#include <stdlib.h>
#include "CUDA_Helper.cuh"



/* Function: DG_QuadLine
 * purpose : get the quad points intergrate to order of input Oder
 * inputs  : Order == order of accuracy 
 * output  : pnq   == pointer to the number of quad points  
 *           pxq   == pointer to the coords of the quad points, 1d here 
 *           qwp   == pointer to the qeights of the quad points
 * return  : cuda error 
 */

cudaError_t 
DG_QuadLine( int Order, int *pnq, double **pxq, double **pwq);



/* Function: DG_Triangle 
 * purpose : get 2d quad points intergrate to order of input Oder
 * inputs  : Order == order of accuracy 
 * output  : pnq   == pointer to the number of quad points  
 *           pxq   == pointer to the coords of the quad points, 2d here 
 *           qwp   == pointer to the qeights of the quad points
 * return  : cuda error 
 */

cudaError_t 
DG_QuadTriangle(int Order, int *pnq, double **pxq, double **pwq);


#endif
