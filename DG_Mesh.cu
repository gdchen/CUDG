/* This is the source file contains the methods for the Mesh struct 
 *
 * Author: Guodong Chen
 * Email: cgderic@umich.edu 
 * Last modified: 12/04/2019
 */

#include "DG_Mesh.cuh"
#include "CUDA_Helper.cuh"

/* initialize the mesh struct */
cudaError_t initMesh(DG_Mesh *Mesh){
 Mesh->halfL = 0;
 Mesh->N = 0;
 Mesh->h = 0;

 Mesh->nNode = 0;     
 Mesh->coord = NULL; // V matrix 

 Mesh->nElem = 0;    
 Mesh->E2N = NULL; // E2N

 Mesh->nIFace = 0;
 Mesh->IFace = NULL;

 Mesh->Jac = NULL;
 Mesh->detJ = NULL;
 Mesh->InvJac = NULL;
 Mesh->Length = NULL;
 Mesh->normal = NULL; 
 return cudaSuccess; 

}


/* create Mesh struct: allocate and initialize */
cudaError_t createMesh(DG_Mesh **pMesh)
{

  CUDA_CALL(cudaMallocManaged(pMesh, sizeof(DG_Mesh))); 
  CUDA_CALL(initMesh(*pMesh));

  return cudaSuccess; 
}


/* actually generate the mesh, fill in the mesh struct members */
cudaError_t generateMesh(DG_Mesh *Mesh, double halfL, int N)
{
  int i, j;
  int counter; 
  Mesh->halfL = halfL; 
  Mesh->N = N;
  Mesh->h = (2*halfL)/N;
  Mesh->nNode = (N+1)*(N+1);
   
  // allocate the coords 
  CUDA_CALL(cudaMallocManaged(&(Mesh->coord), Mesh->nNode*sizeof(double *)));  
  for (i=0; i<Mesh->nNode; i++) 
    CUDA_CALL(cudaMallocManaged(&(Mesh->coord[i]), 2*sizeof(double))); 

  // assign coords for every node 
  for (i=0; i<N+1; i++) 
  {
    for (j=0; j<N+1; j++)
    {
      Mesh->coord[i*(N+1)+j][0] = -halfL + j*Mesh->h;
      Mesh->coord[i*(N+1)+j][1] = -halfL + i*Mesh->h;
    }
  }

  Mesh->nElem = 2*N*N;
  // allocate E2N matrix 
  CUDA_CALL(cudaMallocManaged(&(Mesh->E2N), Mesh->nElem*sizeof(int *))); 
  for (i=0; i<Mesh->nElem; i++) 
    CUDA_CALL(cudaMallocManaged(&(Mesh->E2N[i]), 3*sizeof(int))); 

  // fill in the E2N matrix 
  counter = 0;
  for (i=0; i<N; i++)
  {
    for (j=0; j<N; j++)
    {
      Mesh->E2N[counter][0] = i*(N+1)+j;
      Mesh->E2N[counter][1] = i*(N+1)+j  +1;
      Mesh->E2N[counter][2] = i*(N+1)+j  +(N+1);
      counter++;
      Mesh->E2N[counter][0] = i*(N+1)+j  +1  +(N+1);
      Mesh->E2N[counter][1] = i*(N+1)+j  +1  +N;  
      Mesh->E2N[counter][2] = i*(N+1)+j  +1; 
      counter++;
    }
  }


  Mesh->nIFace = 3*N*N;
  // allocate the interior faces 
  CUDA_CALL(cudaMallocManaged(&(Mesh->IFace), Mesh->nIFace*sizeof(DG_IFace))); 

  counter = 0;
  for (i=0; i<N; i++)
  {
    for (j=0; j<N; j++)
    {
      Mesh->IFace[counter].ElemL = i*(2*N)+j*2; 
      Mesh->IFace[counter].ElemR = i*(2*N)+j*2  +1;
      Mesh->IFace[counter].EdgeL = 0;
      Mesh->IFace[counter].EdgeR = 0;
      Mesh->IFace[counter].node[0] = i*(N+1)+j  +1;
      Mesh->IFace[counter].node[1] = i*(N+1)+j  +(N+1);
      
      counter ++; 
      Mesh->IFace[counter].ElemL = i*(2*N)+j*2;
      if (j==0) Mesh->IFace[counter].ElemR = i*(2*N)+j*2  +2*N-1; // Periodic boundary 
      else  Mesh->IFace[counter].ElemR = i*(2*N)+j*2  -1;
      Mesh->IFace[counter].EdgeL = 1;
      Mesh->IFace[counter].EdgeR = 1;
      Mesh->IFace[counter].node[0] = i*(N+1)+j  +(N+1);
      Mesh->IFace[counter].node[1] = i*(N+1)+j;

      counter++;
      Mesh->IFace[counter].ElemL = i*(2*N)+j*2;
      if (i==0) Mesh->IFace[counter].ElemR = i*(2*N)+j*2  +(N-1)*(2*N)  +1;  // Periodic boundary 
      else Mesh->IFace[counter].ElemR = i*(2*N)+j*2  -(2*N-1);
      Mesh->IFace[counter].EdgeL = 2;
      Mesh->IFace[counter].EdgeR = 2;
      Mesh->IFace[counter].node[0] = i*(N+1)+j; 
      Mesh->IFace[counter].node[1] = i*(N+1)+j  +1;
      counter ++;

    }
  }

  return cudaSuccess; 

}




/* Compute mesh info, include element joacobian, edge length, edge normal */
cudaError_t computeMeshInfo(DG_Mesh *Mesh)
{
  int nElem = Mesh->nElem; 
  int nIFace = Mesh->nIFace;
  DG_IFace *IFace = Mesh->IFace; 
  double **coord = Mesh->coord;
  int **E2N = Mesh->E2N; 

  // allocate the memory for mesh info 
  CUDA_CALL(cudaMallocManaged(&(Mesh->Jac),    nElem*sizeof(double *)));
  CUDA_CALL(cudaMallocManaged(&(Mesh->detJ),   nElem*sizeof(double))); 
  CUDA_CALL(cudaMallocManaged(&(Mesh->InvJac), nElem*sizeof(double *))); 
  CUDA_CALL(cudaMallocManaged(&(Mesh->Length), nIFace*sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&(Mesh->normal), nIFace*2*sizeof(double))); 

  int i;
  double *x0, *x1, *x2;  
  for (i=0; i<nElem; i++){
    // allocate Jacobian data 
    CUDA_CALL(cudaMallocManaged(&(Mesh->Jac[i]), 4*sizeof(double))); 
    // allocate Inverse Jacbian data 
    CUDA_CALL(cudaMallocManaged(&(Mesh->InvJac[i]), 4*sizeof(double))); 

    x0 = coord[E2N[i][0]];
    x1 = coord[E2N[i][1]];
    x2 = coord[E2N[i][2]];
    Mesh->Jac[i][0] = x1[0] - x0[0]; 
    Mesh->Jac[i][1] = x2[0] - x0[0];
    Mesh->Jac[i][2] = x1[1] - x0[1]; 
    Mesh->Jac[i][3] = x2[1] - x0[1];
    Mesh->detJ[i] = Mesh->Jac[i][0]*Mesh->Jac[i][3] - Mesh->Jac[i][1]*Mesh->Jac[i][2]; 
    Mesh->InvJac[i][0] = x2[1] - x0[1]; 
    Mesh->InvJac[i][1] = x0[0] - x2[0]; 
    Mesh->InvJac[i][2] = x0[1] - x1[1];
    Mesh->InvJac[i][3] = x1[0] - x0[0]; 
  }

  double xA, yA, xB, yB; 
  for (i=0; i<nIFace; i++){
    xA = coord[IFace[i].node[0]][0];
    yA = coord[IFace[i].node[0]][1]; 
    xB = coord[IFace[i].node[1]][0];
    yB = coord[IFace[i].node[1]][1];
    Mesh->Length[i] = sqrt((xA-xB)*(xA-xB) + (yA-yB)*(yA-yB));
    Mesh->normal[i*2] = (yB-yA)/(Mesh->Length[i]);
    Mesh->normal[i*2+1] = (xA-xB)/(Mesh->Length[i]); 
  }


  return cudaSuccess; 

}



/* free the mesh memory */
cudaError_t freeMesh(DG_Mesh *Mesh)
{    
  
  int i;

  // free mesh coord 
  for (i=0; i<Mesh->nNode; i++) 
    CUDA_CALL(cudaFree(Mesh->coord[i]));
  CUDA_CALL(cudaFree(Mesh->coord)); 
  
  // free mesh E2N 
  for (i=0; i<Mesh->nElem; i++) 
    CUDA_CALL(cudaFree(Mesh->E2N[i]));
  CUDA_CALL(cudaFree(Mesh->E2N));

  // free interior faces 
  CUDA_CALL(cudaFree(Mesh->IFace));

  // free Jacobian data 
  if (Mesh->Jac != NULL){
    for (i=0; i<Mesh->nElem; i++){
      CUDA_CALL(cudaFree(Mesh->Jac[i]));
      CUDA_CALL(cudaFree(Mesh->InvJac[i]));
    }
    CUDA_CALL(cudaFree(Mesh->Jac)); 
    CUDA_CALL(cudaFree(Mesh->detJ)); 
    CUDA_CALL(cudaFree(Mesh->InvJac));
  }

  // free face length and normal data 
  if (Mesh->Length != NULL) {
    CUDA_CALL(cudaFree(Mesh->Length)); 
    CUDA_CALL(cudaFree(Mesh->normal));
  }

  CUDA_CALL(cudaFree(Mesh));
  return cudaSuccess; 

}


