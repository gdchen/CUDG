/* This is the header file for the DG_All structs, containing all the info for
 * the simulation data 
 * 
 * Author: Guodong Chen
 * Email: cgderic@umich.edu
 * Last modified: 12/06/2019
 */ 

#ifndef _DG_All_H
#define _DG_All_H

#include <stdlib.h>
#include "CUDA_Helper.cuh"
#include "DG_Mesh.cuh"
#include "DG_Basis.cuh"
#include "DG_DataSet.cuh"

/* All structure contains all the structure needed in the DG solver */ 
typedef struct DG_All
{
  DG_Mesh *Mesh; 
  DG_BasisData *BasisData;
  DG_DataSet *DataSet; 
}
DG_All;


/* Function: initAll
 * purpose : initialize the All struct 
 * inputs  : All == pointer to the All struct 
 * outputs : All 
 * return  : error code
 */ 
cudaError_t initAll(DG_All *All); 

/* Function: createAll
 * purpose : allocate and initialize All struct
 * inputs  : pAll  ==  pointer to the All struct pointer
 * outputs : pAll
 * return  : error code
 */ 
cudaError_t createAll(DG_All **pAll);

/* Function: getAllFromIC
 * purpose : fill in the members of All structs, e.g., mesh, basis, states
 * inputs  : All   == pointer to the All struct 
 *           order == approximation order
 *           N     == element in the length and height 
 *           halfL == half of the domain length 
 *          
 * outputs : All  
 * return  : error code
 */
cudaError_t getAllFromIC(DG_All *All, int order, int N, double halfL); 


/* Function: freeAll 
 * purpose : free the memory of the All sturuct 
 * inputs  : All
 * outputs : All
 * return  : error code
 */ 

cudaError_t freeAll(DG_All *All); 


#endif 
