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


// initialization with least square projection 
//void lsqIC(DG_DataSet *DataSet, const DG_Mesh *Mesh, const DG_BasisData *BasisData)
//{
//  int nElem = DataSet->nElem;
//  int np = BasisData->np;
//  int nq2 = BasisData->nq2; 
//  double *Phi = BasisData->Phi;
//  double *wq2 = BasisData->wq2;
//  double *Dwq2 = (double *)malloc(nq2*nq2*sizeof(double));
//  int n, i, j; 
//  for (i=0; i<nq2; i++) {
//    for (j=0; j<nq2; j++) Dwq2[i*nq2+j] = 0;
//    Dwq2[i*nq2+i] = wq2[i]; 
//  }
//  // Initialize State
//  DataSet->State = (double **)malloc(nElem*sizeof(double *));  
//  for (n=0; n<nElem; n++){
//    DataSet->State[n] = (double *)malloc(np*NUM_OF_STATES*sizeof(double));
//    for (i=0; i<np*NUM_OF_STATES; i++) DataSet->State[n][i] = 0; 
//  }
//
//
//  double *b = (double *)malloc(np*NUM_OF_STATES*sizeof(double));
//  for (i=0; i<np*NUM_OF_STATES; i++) b[i] = 0;
//  double *u_exact = (double *)malloc(nq2*NUM_OF_STATES*sizeof(double)); 
//  for (i=0; i<nq2*NUM_OF_STATES; i++) u_exact[i] = 0; 
//  // u_exact 
//  double **xyGlobal = (double **)malloc(nElem*nq2*sizeof(double *));
//  for (i=0; i<nElem*nq2; i++) xyGlobal[i] = (double *)malloc(2*sizeof(double));
//  getGlobalQuadPoints(xyGlobal, Mesh, BasisData);
//  double f0, f1, f2, rho, u, v, p;
//  double *temp = (double *)malloc(np*nq2*sizeof(double));
//  for (i=0; i<np*nq2; i++) temp[i] = 0; 
//  
//  for (i=0; i<nElem; i++){
//    for (j=0; j<nq2; j++){
//      f0 = getf0(xyGlobal[i*nq2+j]);
//      f1 = getf1(f0);
//      f2 = getf2(f0);
//      rho = RHO_INF*pow(f1, 1.0/(GAMMA-1));
//      u = U_INF - f2*(xyGlobal[i*nq2+j][1]-X_ORIGINAL[1]);
//      v = V_INF + f2*(xyGlobal[i*nq2+j][0]-X_ORIGINAL[0]);
//      p = P_INF*pow(f1, GAMMA/(GAMMA-1));
//      u_exact[j*NUM_OF_STATES+0] = rho;
//      u_exact[j*NUM_OF_STATES+1] = rho*u;
//      u_exact[j*NUM_OF_STATES+2] = rho*v;
//      u_exact[j*NUM_OF_STATES+3] = p/(GAMMA-1) + 0.5*rho*(u*u+v*v);
//    }
//    DG_MTxM_Set(np, nq2, nq2, Phi, Dwq2, temp);
//    DG_cMxM_Set(Mesh->detJ[i], np, nq2, NUM_OF_STATES, temp, u_exact, b);  // Integration mapping
//    DG_MxM_Set(np, np, NUM_OF_STATES, DataSet->InvMassMatrix[i], b, DataSet->State[i]);
//  }
//  for (i=0; i<nElem*nq2; i++) free(xyGlobal[i]); free(xyGlobal);
//  free(Dwq2);  free(b); free(u_exact); free(temp);
//
//} 





/* Function:
 * purpose : Get states at elemental interior Quad points 
 * inputs  : State     == elemental state vector 
 *           BasisData == pointer to the BasisData 
 * outputs : Uxy       == states at the quad points 
 *                        should be pre-allocated 
 * return  : cuda error code 
 */  

cudaError_t 
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

cudaError_t 
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
