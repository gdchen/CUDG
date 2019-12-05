All: 

test: testMesh.o 
	nvcc testMesh.o DG_Mesh.o -o testMesh

testMesh: testMesh.c 
	nvcc -c testMesh.c

DG_Mesh: DG_Mesh.cu DG_Mesh.cuh CUDA_Helper.cuh 
	nvcc -c DG_Mesh.cu

clean:
	rm *.o
