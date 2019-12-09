All: test 

test: testMesh.exe testBasis.exe testMath.exe testState.exe testAll.exe

testMesh.exe: ./test/testMesh.cu DG_Mesh.o 
	nvcc -c ./test/testMesh.cu -dc -g
	nvcc testMesh.o DG_Mesh.o -o testMesh.exe -g
	
testBasis.exe: ./test/testBasis.cu DG_Basis.o DG_Quad.o  
	nvcc -c ./test/testBasis.cu -dc -g
	nvcc testBasis.o DG_Basis.o DG_Quad.o -o testBasis.exe -g

testMath.exe: ./test/testMath.cu DG_Math.o 
	nvcc -c ./test/testMath.cu -dc -g
	nvcc testMath.o DG_Math.o -o testMath.exe -g

testState.exe: ./test/testState.cu DG_Mesh.o DG_Quad.o DG_Basis.o DG_Math.o DG_DataSet.o 
	nvcc -c ./test/testState.cu -dc -g
	nvcc testState.o DG_Mesh.o DG_Quad.o DG_Basis.o DG_Math.o DG_DataSet.o -o testState.exe -g

testAll.exe: ./test/testAll.cu DG_Mesh.o DG_Quad.o DG_Basis.o DG_Math.o DG_DataSet.o DG_All.o DG_Residual.o
	nvcc ./test/testAll.cu -dc -g
	nvcc testAll.o DG_Mesh.o DG_Quad.o DG_Basis.o DG_Math.o DG_DataSet.o DG_Residual.o DG_All.o -o testAll.exe -g

DG_Mesh.o: DG_Mesh.cu DG_Mesh.cuh CUDA_Helper.cuh 
	nvcc -c DG_Mesh.cu -dc -g 

DG_Quad.o: DG_Quad.cu DG_Quad.cuh CUDA_Helper.cuh 
	nvcc -c DG_Quad.cu -dc -g

DG_Basis.o: DG_Basis.cu DG_Basis.cuh DG_Quad.cuh CUDA_Helper.cuh 
	nvcc -c DG_Basis.cu -dc -g 

DG_Math.o: DG_Math.cu DG_Math.cuh CUDA_Helper.cuh 
	nvcc -c DG_Math.cu -dc -g 

DG_DataSet.o: DG_DataSet.cu DG_DataSet.cuh DG_Const.cuh DG_Mesh.cuh DG_Basis.cuh DG_Math.cuh 
	nvcc -c DG_DataSet.cu -dc -g

DG_All.o: DG_All.cu DG_All.cuh DG_Mesh.cuh DG_Basis.cuh DG_DataSet.cuh 
	nvcc -c DG_All.cu -dc -g 

DG_Residual.o: DG_Residual.cu DG_Residual.cuh DG_Const.cuh DG_Mesh.cuh DG_Quad.cuh DG_Basis.cuh DG_Math.cuh DG_DataSet.cuh DG_All.cuh 
	nvcc -c DG_Residual.cu -dc -g
clean:
	rm -rf *.txt *.o *.exe
