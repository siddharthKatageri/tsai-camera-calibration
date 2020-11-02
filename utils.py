import numpy as np
from numpy import linalg as LA
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot


def compute_A(data):
    npoints, _ = data.shape
    A = np.zeros((npoints*2, 12))   #initialize A matrix
    first = 0
    sec = 2
    for points in data:
        spam = np.zeros((2,12))

        spam[0,:3] = points[:3]
        spam[0,3] = 1
        spam[0,8:11] = points[:3]*-(points[3])
        spam[0,11] = -points[3]

        spam[1,4:7] = points[:3]
        spam[1,7] = 1
        spam[1,8:11] = points[:3]*-(points[4])
        spam[1,11] = -points[4]

        A[first:sec, :] = spam

        first = sec
        sec = sec+2
    return A

def compute_projection_matrix(A):
    U, s, VT = np.linalg.svd(A)
    V = VT.T
    mask = np.zeros((V.shape[0],1))
    mask[-1] = 1
    projection = np.dot(V,mask)
    return projection


def compute_projection_matrix_hardcode(A):
    AtA = np.dot(A.T,A)
    eigenvalue, eigenvector = LA.eig(AtA)
    pick = np.argmin(eigenvalue)
    mask = np.zeros((eigenvector.shape[0],1))
    mask[pick] = 1
    projection = np.dot(eigenvector,mask)
    return projection


def compute_parameters(projection_matrix):
    Q = projection_matrix[:,:3]
    b = projection_matrix[:,-1]
    b = b.reshape(3,1)

    translation = np.dot((LA.inv(-Q)),b)

    Qinv = np.linalg.inv(Q)
    Rt,Kinv = np.linalg.qr(Qinv)

    intrinsic = np.linalg.inv(Kinv)
    intrinsic_n = intrinsic/intrinsic[2,2]
    #intrinsic_n[0,1] = 0
    rotation = np.transpose(Rt)

    return abs(intrinsic_n), rotation, translation



def reprojection_error(projection_matrix, rotation, translation, data):
    Q = projection_matrix[:,:3]
    Qinv = np.linalg.inv(Q)
    Rt,Kinv = np.linalg.qr(Qinv)
    intrinsic = np.linalg.inv(Kinv)

    f=np.dot(intrinsic, rotation)
    s = np.dot(-Q,translation)
    pcon = np.hstack((f,s))

    wp = data[:,:4].copy()
    wp[:,-1] = 1
    image_points = np.dot(pcon, wp.T).T
    image_points = image_points/image_points[:,2].reshape(-1,1)
    print("\nProjected points are:")
    print(image_points[:,:2])
    error = abs(image_points[:,:2] - data[:,3:])
    return np.mean(error, axis=0)



def visualize_extrinsics(rotation, translation):
    print("\nVisualizing extrinsics now!, computing camera location...")
    xs,ys,zs = [],[],[]
    count=0
    for rotation,translation in zip(rotation, translation):
        mag = 50

        corners = np.array([
                                [-mag,mag,0],
                                [mag,mag,0],
                                [mag,-mag,0],
                                [-mag,-mag,0],
                                [-mag,mag,0],
                                [0,0,mag],
                                [-mag,-mag,0],
                                [0,0,mag],
                                [mag,mag,0],
                                [0,0,mag],
                                [mag,-mag,0],
                                [0,0,mag],
                                [0,0,0],
                                [mag,0,0],
                                [0,0,0],
                                [0,-mag,0],
                                [0,0,0],
                            ])

        #print(corners)
        camera_points = np.dot(corners,rotation)+translation.reshape(1,3) #this give coordinates of camera in world coordinate
        camera_points = np.vstack((camera_points, np.array([0,0,0])))
        print("\ncamera corners in world coordinates are:\n",camera_points)

        '''
        print(np.dot(camera_points-translation.reshape(1,3),rotation.T)) #this is to check camera coordinates
        '''

        x = camera_points[:,0]
        y = camera_points[:,1]
        z = camera_points[:,2]

        xs.append(x)
        ys.append(y)
        zs.append(z)


        fig = go.Figure(data=go.Scatter3d(x=xs[count], y=ys[count], z=zs[count],
                                           #mode='markers',
                                           marker=dict(
                                           size=4,
                                           #color=1,
                                           #colorscale='Viridis',
                                           ),
                                           line=dict(
                                           color='darkblue',
                                           width=2
                                           )
                                ))
        count +=1


    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-10,500],),
                         yaxis = dict(nticks=4, range=[-10,500],),
                         zaxis = dict(nticks=4, range=[-10,500],),),
        margin=dict(r=20, l=10, b=10, t=10))

    fig.show()



def visualize_extrinsics_matplot(rotation, translation):
    print("\nVisualizing extrinsics now!, computing camera location...")
    fig = pyplot.figure()
    ax = Axes3D(fig)
    count=1
    for rotation,translation in zip(rotation, translation):
        mag = 50
        '''
        corners = np.array([[0,0,0],
                            [mag,0,0],
                            [-mag,0,0],
                            [0,mag,0],
                            [0,-mag,0],
                            [0,0,mag],
                            [mag,mag,0],
                            [-mag,mag,0],
                            [mag,-mag,0],
                            [-mag,-mag,0]])
        '''
        corners = np.array([
                                [-mag,mag,0],
                                [mag,mag,0],
                                [mag,-mag,0],
                                [-mag,-mag,0],
                                [-mag,mag,0],
                                [0,0,mag],
                                [-mag,-mag,0],
                                [0,0,mag],
                                [mag,mag,0],
                                [0,0,mag],
                                [mag,-mag,0],
                                [0,0,mag],
                                [0,0,0],
                                [mag,0,0],
                                [0,0,0],
                                [0,-mag,0],
                                [0,0,0],
                            ])


        #print(corners)
        camera_points = np.dot(corners,rotation)+translation.reshape(1,3) #this give coordinates of camera in world coordinate
        camera_points = np.vstack((camera_points, np.array([0,0,0])))
        print("\ncamera corners in world coordinates are:\n",camera_points)

        '''
        print(np.dot(camera_points-translation.reshape(1,3),rotation.T)) #this is to check camera coordinates
        '''

        x = camera_points[:,0]
        y = camera_points[:,1]
        z = camera_points[:,2]



        ax.scatter(x,y,z, color='black', depthshade=False, s=6)
        ax.text(camera_points[5,0], camera_points[5,1], camera_points[5,2], str(count),fontsize=20, color='darkblue')
        ax.set_xlim([-10,500])
        ax.set_ylim([-10,500])
        ax.set_zlim([-10,500])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal','box')
        ax.plot(x, y, z, color='green')
        ax.invert_yaxis()
        count+=1
    return
