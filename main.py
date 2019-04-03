import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import lstsq
from numpy.linalg import qr
from numpy import arctan

def mat_A(X, x):
    '''Build the 2*12 A_i matrix given the x and X vectors'''
    A = np.zeros((2, 12))
    x1,x2,x3 = x[0], x[1], x[2]
    X1,X2,X3,X4 = X[0], X[1], X[2], X[3]
    A[0] = [0, -x3*X1, x2*X1, 0, -x3*X2, x2*X2, 0, -x3*X3, x2*X3, 0, -x3*X4, x2*X4]
    A[1] = [x3*X1, 0, -x1*X1, x3*X2, 0, -x1*X2, x3*X3, 0, -x1*X3, x3*X4, 0, -x1*X4]
    return A

def rq(M):
    '''Compute the RQ decomposition of the matrix M from the numpy.linalg.qr decomposition'''
    n, _ = M.shape
    J = np.fliplr(np.eye(n))
    q, r = qr(np.dot(J, np.dot(M.T, J)))

    return np.dot(J, np.dot(r.T, J)), np.dot(J, np.dot(q.T, J))

def mat_P(p):
    '''Reshape the given p vector of size 12 into the corresponding 3*4 P matrix'''
    matP = np.zeros((3, 4))
    matP[0] = [p[0], p[3], p[6], p[9]]
    matP[1] = [p[1], p[4], p[7], p[10]]
    matP[2] = [p[2], p[5], p[8], p[11]]
    return matP

def mat_K(p):
    '''Compute the matrix K from the vector p of size 12'''
    r, q = rq(mat_P(p)[:, :3])
    return r/r[2, 2]




data = np.loadtxt('data.txt')

X = data[:, :3] # 3D points of the calibration rig
x = data[:, 3:] # Observed 2D points

n, _ = X.shape
ones = np.ones((n, 1))

# Adding an extra 1 coordinate to have normalized points.
X = np.concatenate((X, ones), axis=1)
x = np.concatenate((x, ones), axis=1)

#Plotting input data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.title('3D Points of the calibration rig')
plt.show()

plt.scatter(x[:, 0], x[:, 1], color='b', marker='+', label='Observed')
plt.title('Observed points')
plt.show()


# Building the 2n*12 A matrix by concatening the n A_i matrix
A = np.zeros((0, 12))
for i in range(n):
    A = np.concatenate((A, mat_A(X[i, :], x[i, :])))

# Solving A_11x = b with A_11 the A matrix without the last column and x a vector of size 11
m = lstsq(A[:, :11], -A[:, -1], rcond=None)
p = m[0]
residus = m[1]

# Adding the missing composant, fixed to 1
p = np.append(p, 1)
K = mat_K(p)

print('\nVector p :\n{}'.format(p))
print('\nProjection matrix P :\n{}'.format(mat_P(p)))
print('\nResidues :\n{}'.format(residus[0]))
print('\nCalibration matrix K :\n{}'.format(K))
print('\nPixel\'s base :\n{}'.format(K[0, 0]))
print('\nPixel\'s height :\n{}'.format(K[1, 1]))
print('\nTrapezoïd\'s angle :\n{}\n'.format(arctan(-K[0, 0]/K[0, 1])))


# Retrieving the projection of the observed 3D points thanks to the P matrix
x_approx = np.zeros((n, 3))
P = mat_P(p)
for i in range(n):
    x_approx[i] = np.dot(P, X[i])/np.dot(P, X[i])[2] # Don't forget to normalize the projection

# Plotting observed and computed points
plt.scatter(x[:, 0], x[:, 1], color='b', marker='+', label='Observed')
plt.scatter(x_approx[:, 0], x_approx[:, 1], color='r', marker='+', label='Computed with P')
plt.legend()
plt.title('Observed and computed points')
plt.show()
