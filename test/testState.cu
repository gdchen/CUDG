#include <stdio.h>
#include <stdlib.h>
#include "../DG_DataSet.cuh"
#include "../DG_Mesh.cuh"


int main()
{
  int order = 1;
  int N = 1;
  int np = (order+1)*(order+2)/2; 
  DG_Mesh *Mesh; 
  createMesh(&Mesh);
  generateMesh(Mesh,5,N);
  computeMeshInfo(Mesh);
  
  DG_BasisData *BasisData; 
  createBasisData(&BasisData);
  computeBasisData(order, BasisData);

  DG_DataSet *DataSet; 
  createDataSet(&DataSet);
  computeMassMatrix(DataSet, Mesh, BasisData); 
  int nElem = Mesh->nElem;
  //int np = BasisData->np; 
  int n, i, j;
  double eye[np*np]; 
  for (n=0; n<nElem; n++)
  {
    printf("%dth element mass: \n", n);
    for (i=0; i<np; i++){
      for (j=0; j<np; j++) printf("%f ", DataSet->MassMatrix[n][i*np+j]);
      printf("\n");
    }
    printf("Inver matrix: \n");
    for (i=0; i<np; i++){
      for (j=0; j<np; j++) printf("%f ", DataSet->InvMassMatrix[n][i*np+j]);
      printf("\n");
    }
    DG_MxM_Set(np, np, np, DataSet->MassMatrix[n], DataSet->InvMassMatrix[n], eye); 
    for (i=0; i<np; i++){
      for (j=0; j<np; j++) printf("%f ", eye[i*np+j]);
      printf("\n");
    }
  }







  int nGlobal = nElem*np; 
  double **xyGlobal = (double **)malloc(nGlobal*sizeof(double *));
  for (i=0; i<nGlobal; i++) xyGlobal[i] = (double *)malloc(2*sizeof(double));
  getGlobalLagrangeNodes(order, Mesh, xyGlobal);
/*  for (i=0; i<nGlobal; i++)
    printf("%f %f \n", xyGlobal[i][0], xyGlobal[i][1]);
  printf("%d %d \n", Mesh->nElem, 2*N*N);*/

  // Test interpolation 
  interpolateIC(DataSet, Mesh);
  FILE *coordX, *coordY, *state;
  coordX = fopen("coordX.txt", "w");
  coordY = fopen("coordY.txt", "w");
  state = fopen("state.txt", "w");
  for (i=0; i<nElem; i++){
    for (j=0; j<np; j++){
      fprintf(coordX, "%f ", xyGlobal[i*np+j][0]);
      fprintf(coordY, "%f ", xyGlobal[i*np+j][1]);
      fprintf(state, "%f %f %f %f \n", DataSet->State[i][j*NUM_OF_STATES+0], DataSet->State[i][j*NUM_OF_STATES+1],
                                       DataSet->State[i][j*NUM_OF_STATES+2], DataSet->State[i][j*NUM_OF_STATES+3]);
    }
  }
  fclose(coordX);
  fclose(coordY);
  fclose(state);

  // test least square 
/*  lsqIC(DataSet, Mesh, BasisData);
  FILE *coordX, *coordY, *state;
  coordX = fopen("coordX.txt", "w");
  coordY = fopen("coordY.txt", "w");
  state = fopen("state.txt", "w");
  for (i=0; i<nElem; i++){
    for (j=0; j<np; j++){
      fprintf(coordX, "%f ", xyGlobal[i*np+j][0]);
      fprintf(coordY, "%f ", xyGlobal[i*np+j][1]);
      fprintf(state, "%f %f %f %f \n", DataSet->State[i][j*NUM_OF_STATES+0], DataSet->State[i][j*NUM_OF_STATES+1],
                                       DataSet->State[i][j*NUM_OF_STATES+2], DataSet->State[i][j*NUM_OF_STATES+3]);
    }
  }
  fclose(coordX);
  fclose(coordY);
  fclose(state);*/

  


  for (i=0; i<nGlobal; i++) free(xyGlobal[i]); free(xyGlobal);
  freeBasisData(BasisData);
  freeDataSet(DataSet); 
  freeMesh(Mesh);
  return 0;
}
