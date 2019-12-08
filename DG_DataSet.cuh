#ifndef _DG_DataSet_H_
#define _DG_DataSet_H_


#include <stdlib.h>
#include "DG_Const.cuh"
#include "DG_Mesh.cuh"
#include "DG_Basis.cuh"
#include "DG_Math.cuh"

// DataSet structure stores the states vector and mass matrix 

typedef struct DG_DataSet
{
  int nElem;
  int order; 
  double **State;      // States vector, nElem npxNUM_OF_STATES matrix 
  double **MassMatrix; // Mass matrix, nElem npxnp mass matrix 
  double **InvMassMatrix; // inverse of mass matrix, nElem npxnp matrix 
}
DG_DataSet;


/* Function: initDataSet 
 * purpose : initialize the DataSet struct 
 * inputs  : DataSet == pointer to the DataSet 
 * outputs : DataSet 
 * return  : cuda error code 
 */ 

cudaError_t initDataSet(DG_DataSet *DataSet); 


/* Function: createDataSet 
 * purpose : allocate memory and initialize the DataSet struct 
 * inputs  : pDataSet == pointer to the DataSet pointer
 * outputs : pDataSet 
 * return  : cuda error code 
 */ 

cudaError_t createDataSet(DG_DataSet **pDataSet); 


/* Function: freeDataSet 
 * purpose : free the memory of the dataset  
 * inputs  : DataSet == pointer to the dataset 
 * outputs : none
 * return  : cuda error code 
 */

cudaError_t freeDataSet(DG_DataSet *DataSet); 


/* Function: computeMassMatrix 
 * purpose : compute the mass matrix of using the basis and mesh data 
 * inputs  : DataSet   == pointer to the DataSet struct 
             Mesh      == pointer to the Mesh struct
             BasisData == pointer to the Basis data 
 * outputs : DataSet 
 * return  : cuda error code
 */ 
 
cudaError_t 
computeMassMatrix(DG_DataSet *DataSet, const DG_Mesh *Mesh, const DG_BasisData *BasisData); 


/* Function: DG_InterpolateIC
 * purpose : initialize the state with direct interpolation 
 * inputs  : Dataset == pointer to the DataSet struct  
 *           Mesh    == pointer to the Mesh struct 
 * outputs : DataSet 
 * return  : cuda error code 
 */ 
cudaError_t interpolateIC(DG_DataSet *DataSet, const DG_Mesh *Mesh);


/* Function:
 * purpose : Get states at elemental interior Quad points 
 * inputs  : State     == elemental state vector 
 *           BasisData == pointer to the BasisData 
 * outputs : Uxy       == states at the quad points 
 *                        should be pre-allocated 
 * return  : cuda error code 
 */  

__device__ __host__ cudaError_t 
getIntQuadStates(double *Uxy, const double *State, const DG_BasisData *BasisData);




/* Function: Get states the edge quad points 
 * inputs  : edge      == local edge index, only need one since 
 *                        it's the same for elemL and elemR
 *           StateL    == elemental state vector from elemL 
 *           StateR    == elemental state vector form elemR
 *           BasisData == pointer to the BasisData 
 * outputs : UL        == left elem states at the edge quad points
 *                        should be pre-allocated
 *           UR        == right elem states at the edge quad points
 *                        should be pre-allocated
 * return  : cuda error code  
 */

__device__ __host__ cudaError_t 
getEdgeQuadStates(double *UL, double *UR, int edge, const double *StateL, const double *StateR, 
                  const DG_BasisData *BasisData);



/* Function: getLagrangeNodes 
 * purpose : get reference Lagrange nodes of order [order]
 * inputs  : order  == approx order
 * outputs : xy     == coords of Lagrange nodes 
                       should be pre-allocated 
 * retirn  : cuda error code
 */ 

cudaError_t getLagrangeNodes(int order, double **xy); 



/* Function: getGlobalLagrangeNodes 
 * purpose : get global Lagrange nodes of order [order]
 * inputs  : order        == approx order
 *           Mesh         == pointer to the mesh struct 
 * outputs : xyGlobal     == coords of Lagrange nodes
                             should be pre-allocated 
 * return  : cuda error code
 */ 

cudaError_t getGlobalLagrangeNodes(int order, const DG_Mesh *Mesh, double **xyGlobal);


/* Function: getGlobalQuadNodes 
 * purpose : get global quad nodes of order [order]
 * inputs  : order        == approx order
 *           Mesh         == pointer to the mesh struct 
 *           BasisData    == pointer to the BasisData
 * outputs : xyGlobal     == coords of Lagrange nodes
                             should be pre-allocated 
 * return  : cuda error code
 */ 

cudaError_t 
getGlobalQuadPoints(double **xyGlobal, const DG_Mesh *Mesh, const DG_BasisData *BasisData);





/*Functions of exact solution*/
double getf0(double *x); 
double getf1(double f0); 
double getf2(double f0);


#endif 
