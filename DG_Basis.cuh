/* This file is the header file for the definition of the DG_Basis structs 
 * and the function prototypes 
 * 
 * Author: Guodong Chen
 * Email : cgderic@umich.edu
 * Last modified: 12/05/2019
 */


#ifndef _DG_Basis_
#define _DG_Basis_

#include <stdlib.h>
#include "DG_Quad.cuh"
#include "CUDA_Helper.cuh" 


/* BasisData contains all the basis info */ 
typedef struct DG_BasisData
{
  int order; // order of solution 
  int np;    // # of basis functions  
  /* Edge Quad*/
  int nq1;      // # of 1d quad points 
  double *sq;   // 1d Quad points 
  double *wq1;  // Weights 
  double **EdgePhiL; // left elem basis fcns at 1d points 
  double **EdgePhiR; // right elem basis fcns at 1d points 

  /* Interior 2d Quad*/
  int nq2;   // # of 2d quad points 
  double *xyq;    // 2d quad points
  double *wq2;    // weights 
  double *Phi;    // basis fcns at 2d quad points 
  double *GPhix;  // gradx of basis fcns at 2d quad points
  double *GPhiy;  // grady of basis fcns at 2d quad points 
}
DG_BasisData;



/* Function: initBasisData 
 * purpose : initialize the Basis Data
 * inputs  : BasisData == pointer to the BasisData
 * outputs : BasisData
 * return  : cudaError
 */ 
 
cudaError_t initBasisData(DG_BasisData *BasisData); 



/* Function: createBasisData 
 * purpose : allocates the Basis Data and initialize it 
 * inputs  : BasisData == pointer to the BasisData
 * outputs : BasisData
 * return  : cudaError
 */ 

cudaError_t createBasisData(DG_BasisData **pBasisData); 



/* Function: freeBasisData 
 * purpose : free the memory of the basis data
 * inputs  : BasisData == pointer to the BasisData
 * outputs : BasisData
 * return  : cudaError
 */ 

cudaError_t freeBasisData(DG_BasisData *BasisData); 


/* Function: computeBasisData
 * purpose : compute the basis data for interior quad points and 
 *           edge quad points, all the members get allocated here
 * inputs  : p == elemental approximation order '
 *         : BasisData == pointer to the BasisData 
 * outputs : BasisData
 * return  : cuda error code
 */ 

cudaError_t computeBasisData(int p, DG_BasisData *BasisData);

/* Function: RefEdge2Elem 
 * purpose : convert edge coords to elem coords 
 * inputs  : edge == local edge index
 *           sq   == edge coords 
 * outputs : xy   == elem coords
 * return  : 0 
 */ 

cudaError_t RefEdge2Elem(const int edge, double *xy, const double sq); 



/* Function: DG_TriLagrange
   purpose : basis function evaluated at input elem coords 
   inputs  : p   == Basis order 
             xy  == elem coords 
   outputs : Phi == basis function evaluated at xy 
   return  : 0
 */ 
cudaError_t DG_TriLagrange(int p, const double *xy, double *phi); 


/* Function: DG_Grad_TriLagrange
   purpose : basis function grads evaluated at input elem coords 
   inputs  : p   == Basis order 
             xy  == elem coords 
   outputs : gPhi == basis function grads evaluated at xy 
   return  : 0
 */ 
cudaError_t DG_Grad_TriLagrange(int p, const double *xy, double *gphi);



#endif 
