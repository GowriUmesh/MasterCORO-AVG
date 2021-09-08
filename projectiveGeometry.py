'''
Author: Minh Quan Dao
Version: 1.0

Modifier: Gowri Umesh <mailgowriumesh@gmail.com>,
          R Singaram <singaram888@gmail.com>

Description: This script takes an image with perscpective projecion
and performs affine rectificaiton and metric rectificaiton o that image
thereby retaining the parallelism and orthonognality of the image.
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

def euclidean_trans(theta, tx, ty):
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty],
        [0, 0, 1]
    ])
def rot_x(theta):
    return np.array([
        [1,0,0],
        [0,np.cos(theta), -np.sin(theta)],
        [0,np.sin(theta), np.cos(theta)]
    ])

filename = "img/fig1_6c.jpg"
img = cv2.imread(filename)
plt.imshow(img)
#plt.show()

'select initial 4 points which are parallel in the world frame'
points = np.array([
    [1053.01298701,  460.10606061],
    [1364.7012987,   641.92424242],
    [1057.34199134,  867.03246753],
    [745.65367965,  641.92424242]
])

#print('chosen coordinates: \n', points)  # each row is a point

plt.plot(*zip(*points), marker='o', color='r', ls='')
plt.imshow(img)
plt.savefig('Results/originalImage.png')
#plt.show()

'''
Affine rectification: 0
'''
print('\n<-------- Task 1 :Affine rectification -------->')
pts_homo = np.concatenate((points, np.ones((4, 1))), axis=1)
# convert chosen pts to homogeneous coordinate
#print('Task 1.1: Identify image of the line at inf on projective plane')
hor_0 = np.cross(pts_homo[0,],pts_homo[1]) # the first horizontal line
hor_1 = np.cross(pts_homo[2],pts_homo[3]) #
pt_ideal_0 = np.cross(hor_0,hor_1)# first ideal point
pt_ideal_0 /= pt_ideal_0[-1]  # normalize
#print('@Task 1.1: first ideal point: ', pt_ideal_0)

ver_0 = np.cross(pts_homo[0],pts_homo[3])# the 1st vertical line
ver_1 = np.cross(pts_homo[1],pts_homo[2])# the 2nd vertical line
pt_ideal_1 = np.cross(ver_0,ver_1)# 2nd ieal point
pt_ideal_1 /= pt_ideal_1[-1]
#print('@Task 1.1: second ideal point: ', pt_ideal_1)

l_inf = np.cross(pt_ideal_0,pt_ideal_1)# image of line at inf
l_inf /= l_inf[-1]
#print('@Task1.1: line at infinity: ', l_inf)

#print('Task 1.2: Construct the projectivity that affinely rectify image')
H = np.array([
    (1,0,0),
    (0,1,0),
    (l_inf)
])
#print('@Task 1.2: image of line at inf on affinely rectified image: ', (np.linalg.inv(H).T @ l_inf.reshape(-1, 1)).squeeze())

H_E = euclidean_trans(np.deg2rad(0),50,250)

affine_img = cv2.warpPerspective(img, H_E @ H, (img.shape[1], img.shape[0]))
new_homo_pts = H_E @ H  @ pts_homo.T
affine_pts = new_homo_pts.T 
for i in range(affine_pts.shape[0]):
    affine_pts[i] /= affine_pts[i, -1]

plt.plot(*zip(*affine_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(affine_img)
plt.savefig('Results/AffineRectifiedImage.png')
#plt.show()
print('<-------- Saving Affine Rectified Image -------->')
print('<-------- End of Task 1 -------->\n')



'''
Task 2: Metric rectification
'''
print('\n<-------- Task 2: Metric rectification -------->')
#print('Task 2.1: transform 4 chosen points from projective image to affine image')
aff_hor_0 =np.cross(affine_pts[0],affine_pts[1])# image of first horizontal line on affine plane
aff_hor_1 =np.cross(affine_pts[2],affine_pts[3]) # image of 2nd horizontal line on affine plane

aff_ver_0 = np.cross(affine_pts[0],affine_pts[3])# image of first vertical line on affine plane
aff_ver_1 = np.cross(affine_pts[1],affine_pts[2])# image of 2nd vertical line on affine plane

diag_0 = np.cross(affine_pts[0],affine_pts[2])
diag_1 = np.cross(affine_pts[1],affine_pts[3])

aff_hor_0 /= aff_hor_0[-1]
aff_hor_1 /= aff_hor_1[-1]
aff_ver_0 /= aff_ver_0[-1]
aff_ver_1 /= aff_ver_1[-1]
diag_0 /=diag_0[-1]
diag_1 /=diag_1[-1]
'''
print('@Task 2.1: first chosen point coordinate')
print('\t\t on projective image: ', pts_homo[0])
print('\t\t on affine image: ', affine_pts[0])
print('Task 2.2: construct constraint matrix C to find vector s')
'''
C0 = np.array([
    (aff_hor_0[0]*aff_ver_0[0] ,aff_hor_0[0]*aff_ver_0[1] + aff_hor_0[1]*aff_ver_0[0] ,aff_hor_0[1]*aff_ver_0[1])

])

C1 = np.array([
    (
    diag_0[0] * diag_1[0],diag_0[0] * diag_1[1] + diag_1[1] * diag_0[0], diag_0[1] * diag_1[1])
])

C = np.vstack([C0, C1])
'''
print(np.linalg.matrix_rank(C))
print('@Task 2.2: constraint matrix C:\n', C)
print('Task 2.3: Find s by looking for the kernel of C (hint: SVD)')
'''
U,a,vh = np.linalg.svd(C)
s = vh.T[:,-1]
'''
print('@Task 2.3: s = ', s)
print('@Task 2.3: C @ s = \n', C @ s.reshape(-1, 1))
'''
mat_S = np.array([
    [s[0], s[1]],
    [s[1], s[2]],
])
'''
print('@Task 2.3: matrix S:\n', mat_S)
print('Task 2.4: Find the projectivity that do metric rectificaiton')
'''
eigvals,eigvecs = np.linalg.eig(mat_S)
K = eigvecs @ np.sqrt(np.diag(eigvals))
Kinv = np.linalg.inv(K)

H = np.array([
    (Kinv[0,0],Kinv[0,1],0),
    (Kinv[1,0],Kinv[1,1],0),
    (0,0,1)
])

aff_dual_conic = np.array([
    [s[0], s[1], 0],
    [s[1], s[2], 0],
    [0, 0, 0]
])

#print('@Task 2.5: image of dual conic on metric rectified image: \n', H @ aff_dual_conic @ H.T)

H_E =  euclidean_trans(np.deg2rad(0),0,-600)@ euclidean_trans(np.deg2rad(-13),0,0)
H_fin = H_E @ H
eucl_img = cv2.warpPerspective(affine_img, H_fin, (img.shape[1], img.shape[0]))

eucl_pts = (H_fin  @ affine_pts.T).T
#print(eucl_pts)
for i in range(eucl_pts.shape[0]):
    eucl_pts[i] /= eucl_pts[i, -1]

plt.plot(*zip(*eucl_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(eucl_img)
plt.savefig('Results/euclideanImage.png')
print('<-------- Saving Metric Rectified Image -------->')
print('<-------- End of Task 2 -------->')
#plt.show()
