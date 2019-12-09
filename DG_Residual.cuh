/* This is the header file for DG_Residual, containing the function prototypes 
 * for residual evaluations in DG solve 
 *
 * Author: Guodong Chen
 * Email: cgderic@umich.edu
 * Last modified: 12/07/2019
 */

#ifndef _DG_Residual_H
#define _DG_Residual_H


#include <stdlib.h>
#include "DG_Mesh.cuh"
#include "DG_Quad.cuh"
#include "DG_Basis.cuh"
#include "DG_DataSet.cuh"
#include "DG_All.cuh"


/* Function: calculateFlux 
 * purpose: F(U) function in the residual evaluation, 
 * inputs : nq2    ==   # of 2D interior quad points
 *          Uxy    ==   interior states at quad points (nq*NUM_OF_STATES)
 * outputs: 
 *          Fx     ==   x Flux at quad points
 *          Fy     ==   y Flux at quad points 
 * return : error code 
 */ 
__device__  
int calculateFlux(double *Fx, double *Fy, int nq2, const double *Uxy);



/* Function: getFhat 
 * purpose : Fhat function (numerical flux) in the residual evaluation 
 *           at a single quad point 
 * inputs  : 
 *           UL        ==    states from elemL at current 1D quad point 
 *           UR        ==    sattes from elemR at current 1D quad point 
 *           normal    ==    normal vector, elemL ==> elemR
 * outputs :
 *           Fhat      ==    numerical flux, Roe flux is used here 
 *
 * return  : error code
 */ 

__device__ 
int getFhat(double *Fhat, const double *UL, const double *UR, const double *normal); 


/* Function: calculateFhat
 * purpose : calculate the roe flux along the edge, all the 1D quad points
 * inputs  : 
 *            nq1    ==   # of 1D quad points 
 *            UL     ==   states from elemL along the edge 
 *            UR     ==   states from elemR along the edge
 *            normal ==   normal vector, elemL ==> elemR
 * outputs :  Fhat   ==   roe fluxes along the edge
 * return  :  error code
 */ 
__device__ 
int calculateFhat(double *Fhat, int nq1, const double *UL, const double *UR, const double *normal);


/* Kernel Function: calculateVolumeRes
 * purpose : calculate the volumes residual contribution 
 * inputs  : All      ==   All struct
 * outputs : State    ==   temp state vector, used to store intermediate 
 *                         state vector for time intergration 
 *           R        ==   Residual term 
 * return  : error code 
 */
__global__ void 
calculateVolumeRes(const DG_All *All, double **State, double **R); 

/* Kernel Function: calculateFaceRes
 * purpose :  All     ==  all structs 
 * inputs  :  States  ==  temp state vector, used to store intermediate
 *                        state vector for time integration 
 * outputs :  Rf      ==  Face residual vector 
 * return  :  error code 
 */

__global__ void 
calculateFaceRes(const DG_All *All, double **State, double **RfL, double **RfR); 


/* Kernel Function: addRes
 * purpose :
 * inputs  :
 * outputs :
 * return  :
 */
__global__ void  
addRes(const DG_All *All, double **R, double **RfL, double **RfR);


__global__ void
Res2RHS(const DG_All *All, double **R, double **f); 


__global__ void 
rk4_inter(DG_All *All, double **State, double dt, double **f); 


__global__ void 
rk4_final(DG_All *All, double dt, double **f0, double **f1, double **f2, double **f3); 


cudaError_t 
DG_RK4(DG_All *All, double dt, int Nt); 


#endif 
