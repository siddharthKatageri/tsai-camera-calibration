import numpy as np
import utils
from numpy import linalg as LA
from scipy.linalg import svd
import cv2
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import os

root = './data_points/'
rotation_matrices, translation_vectors = [],[]

for file in os.listdir(root):
    print(file)
    data = np.loadtxt(root+file)
    print(data)
    npoints, _ = data.shape
    print("number of points are:", npoints)

    A = utils.compute_A(data)
    #print(A)

    projection_matrix = utils.compute_projection_matrix(A)
    projection_matrix = projection_matrix.reshape(3,4)
    print("\nprojection matrix :\n",projection_matrix)

    #K, R, t = cv2.decomposeProjectionMatrix(projection_matrix)[:3]


    intrinsic, rotation, translation = utils.compute_parameters(projection_matrix)
    rotation_matrices.append(rotation)
    translation_vectors.append(translation)

    print("\nintrinsics:\n", intrinsic)
    print("\nrotation:\n", rotation)
    print("\ntranslation:\n", translation)


    error = utils.reprojection_error(projection_matrix, rotation, translation, data)
    print("\nReprojection error is:",error)


    #a small example
    '''
    p = np.array([[1],[2],[3]])    #this point is in the camera coordinate
    tp = np.dot(rotation, p)+translation #this line gives me the world coordinate of the point
    print(p)
    print(tp)
    print(np.dot(rotation.T, tp-translation))   #this lines takes me back to the camera coordinate
    '''

utils.visualize_extrinsics_matplot(rotation_matrices, translation_vectors) #this function used matplotlib


pyplot.show()
