# This is a python script visiualize the mesh 
# Author: Guodong Chen
# Email: cgderic@umich.edu
# Last modified: 12/05/2019

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# input mesh file shoud be named as E.txt and V.txt 
# E is the E2N matrix, V is the coord of the nodes 
E = np.genfromtxt('E.txt', dtype =int)
V = np.genfromtxt('V.txt')
halfL = max(V[:,0]);

plt.figure()
for k in range(len(E)):
    N = E[k,:]
    x = V[N,0]
    y = V[N,1]
    x = np.append(x,x[0])  # need to go back to the x0 to close the triangle
    y = np.append(y,y[0])
    plt.plot(x,y,'k-');

plt.gca().set_aspect('equal')
plt.axis([-halfL, halfL, -halfL, halfL])
plt.title('Mesh (N=%d)'%(np.sqrt(len(V))-1))
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([-halfL, 0, halfL])
plt.yticks([-halfL, 0, halfL])
plt.tight_layout
plt.savefig("Mesh.pdf",bbox_inches='tight')
#plt.show()
