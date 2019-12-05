#include <stdio.h>
#include "../CUDA_Helper.cuh"
#include "../DG_Mesh.cuh"

int main()
{
  DG_Mesh *Mesh;
  createMesh(&Mesh); 
  int N = 8;
  generateMesh(Mesh,5,N);
  printf("halfL %f,  N %d,  h %f \n", Mesh->halfL,Mesh->N, Mesh->h);
  printf("nNode %d \n", Mesh->nNode);

  FILE *V, *E;
  V = fopen("V.txt", "w");
  E = fopen("E.txt", "w"); 
  int i ,j;
  for (i=0; i<Mesh->nNode; i++) 
    fprintf(V, "%f %f \n", Mesh->coord[i][0], Mesh->coord[i][1]);
  printf("nElem %d\n", Mesh->nElem);
  for (i=0; i<Mesh->nElem; i++)
    //printf("%d %d %d \n", Mesh->E2N[i][0], Mesh->E2N[i][1], Mesh->E2N[i][2]);
    fprintf(E, "%d %d %d \n", Mesh->E2N[i][0], Mesh->E2N[i][1], Mesh->E2N[i][2]);
  printf("nIFace %d \n", Mesh->nIFace);
  fclose(V);
  fclose(E);
  DG_IFace  *IFace = Mesh->IFace;
  for (i=0; i<Mesh->nIFace; i++)
    printf("%d %d %d %d \n", IFace[i].ElemL, IFace[i].ElemR, IFace[i].node[0], IFace[i].node[1]);
  computeMeshInfo(Mesh);
  printf("*************************************\n");
  printf("Jacobian Info \n");
  printf("*************************************\n");
  for (i=0; i<Mesh->nElem; i++){
    printf("%dth element: \n", i);
    printf("Jacobian: \n");
    for (j=0; j<2; j++){
      printf("%f %f \n", Mesh->Jac[i][2*j], Mesh->Jac[i][2*j+1]);
    }
    printf("detJ is %f \n", Mesh->detJ[i]);
    for (j=0; j<2; j++){
      printf("%f %f \n", Mesh->InvJac[i][2*j], Mesh->InvJac[i][2*j+1]);
    }
  }
  printf("****************************************\n");
  printf("Length and normal :\n");
  printf("****************************************\n");
  for (i=0; i<Mesh->nIFace; i++)
  {
    printf("%dth face: \n", i);
    printf("Legth %f \n", Mesh->Length[i]);
    printf("Normal: %f %f \n", Mesh->normal[i*2], Mesh->normal[i*2+1]);
  }
  freeMesh(Mesh);








  return 0;
}
