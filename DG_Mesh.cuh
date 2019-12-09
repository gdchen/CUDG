/* This is the header file contains the Mesh Struct for DG FEM, and function 
 * prototypes for its methods. 
 * 
 * Author: Guodong Chen
 * Email: cgderic@umich.edu
 * Last modified: 12/04/2019 
 *
 */


#ifndef _DG_MESHSTRUCT_
#define _DG_MESHSTRUCT_ 

#include <stdlib.h>
#include <math.h>
#include "CUDA_Helper.cuh"

/* Interior face struct  */
typedef struct DG_IFace 
{
  int ElemL;
  int ElemR;      // left/right elem
  int EdgeL;
  int EdgeR;      // local edge number of left/right elem 
  int node[2];    // node index of the egde
}
DG_IFace;


/* Mesh struct */
typedef struct DG_Mesh
{
  // physical information 
  double halfL; // half of the domian length 
  int N; // number of intervals in the x and y direction 
  double h; //spacing 

  // Node information 
  int nNode;  // number of nodes 
  double **coord; // cooordinates of the nodes 

  // Element info 
  int nElem;  // number of elemnets 
  int **E2N; // E2N matrix (hash table)
 

  // Face information
  int nIFace; 
  int **E2F; 
  DG_IFace *IFace;

  // Jacobian infomation 
  double **Jac; 
  double *detJ;
  double **InvJac;  // Actually stores detJ*InvJac for convenience 
  double *Length;   // Index by faces  
  double *normal;   // Index by faces 

   
}
DG_Mesh;



/* Function: initMesh
 * purpose : initialize members of Mesh struct 
 * inputs  : Mesh == pointer to the mesh struct
 * outputs : Mesh
 * return  : cuda error
 */

cudaError_t initMesh(DG_Mesh *Mesh); 



/* Function: createMesh 
 * purpose : allocate the memory for Mesh struct and initialize it 
 * inputs  : pMesh == pointer to the Mesh pointer
 * outputs : pMesh 
 * return  : cuda error
 */

cudaError_t createMesh(DG_Mesh **pMesh);



/* Function: genMesh
 * purpose : generate the mesh struct, fill in the members, i.e., mesh connectivity  
 * inputs  : Mesh  == pointer to the mesh struct 
 *           halfL == half of the domain length  
 *           N     == number of intervals on each side
 * outputs : Mesh  == mesh struct with members assigned 
 * return  : cuda error
 */

cudaError_t generateMesh(DG_Mesh *Mesh, double halfL, int N); 



/* Function: computeMeshInfo
 * purpose : precompute the mesh info, including element jacobian, edge normals 
 * inputs  : Mesh == pointer to the mesh struct 
 * outputs : Mesh 
 * return  : cuda error
 */

cudaError_t computeMeshInfo(DG_Mesh *Mesh);



/* Function: freeMesh 
 * purpose : free the mesh memory 
 * inputs  : Mesh == pointer to the mesh struct  
 * outputs : Mesh
 * return  : cuda error
 */
cudaError_t freeMesh(DG_Mesh *Mesh);


#endif 
