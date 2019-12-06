All: test 

test: testMesh.exe testBasis.exe testMath.exe

testMesh.exe: ./test/testMesh.cu DG_Mesh.o 
	nvcc -c ./test/testMesh.cu 
	nvcc testMesh.o DG_Mesh.o -o testMesh.exe 
	
testBasis.exe: ./test/testBasis.cu DG_Basis.o DG_Quad.o  
	nvcc -c ./test/testBasis.cu 
	nvcc testBasis.o DG_Basis.o DG_Quad.o -o testBasis.exe 

testMath.exe: ./test/testMath.cu DG_Math.o 
	nvcc -c ./test/testMath.cu 
	nvcc testMath.o DG_Math.o -o testMath.exe

DG_Mesh.o: DG_Mesh.cu DG_Mesh.cuh CUDA_Helper.cuh 
	nvcc -c DG_Mesh.cu

DG_Quad.o: DG_Quad.cu DG_Quad.cuh CUDA_Helper.cuh 
	nvcc -c DG_Quad.cu

DG_Basis.o: DG_Basis.cu DG_Basis.cuh DG_Quad.cuh CUDA_Helper.cuh 
	nvcc -c DG_Basis.cu 

DG_Math.o: DG_Math.cu DG_Math.cuh CUDA_Helper.cuh 
	nvcc -c DG_Math.cu
#src: DG_Mesh DG_Quad DG_Basis 

clean:
	rm -rf *.o *.exe
