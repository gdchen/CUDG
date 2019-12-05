All: 

test: testMesh 
	nvcc testMesh.o DG_Mesh.o -o testMesh.exe

testMesh: ./test/testMesh.cu 
	nvcc -c ./test/testMesh.cu 


DG_Mesh: DG_Mesh.cu DG_Mesh.cuh CUDA_Helper.cuh 
	nvcc -c DG_Mesh.cu

clean:
	rm *.o
