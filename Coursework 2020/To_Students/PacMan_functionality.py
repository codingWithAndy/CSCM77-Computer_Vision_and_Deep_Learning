# PacMan tools
# The following are some auxillery functions for playing the 3D pointcloud PacMan game in Python. 
# Take a moment to read through the functions and get a feel for their purpose.
# Mike Edwards, 2019

# Usual imports
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [9,9]
from itertools import permutations, repeat

global_cloud = {}
global_cloud['Positions'] = np.load('cloudPositions.npy')
global_cloud['Colors'] = np.load('cloudColors.npy')


def sphere(n):
    """
    Defines the XYZ coordinates for n points across a unit sphere.
    
    Input: 
            n - Number of points to sample.
    
    Output: 
            coords - XYZ locations of points across the sphere.
    """
    
    theta = np.arange(-n, n + 1, 2) / n * np.pi
    phi =  np.arange(-n, n + 1, 2).T / n * np.pi / 2
    cosphi = np.cos(phi)
    cosphi[0] = 0
    cosphi[n] = 0
    sintheta = np.sin(theta) 
    sintheta[0] = 0 
    sintheta[n] = 0
    x = (cosphi * np.expand_dims(np.cos(theta), axis=1)).flatten()
    y = (cosphi * np.expand_dims(sintheta, axis=1)).flatten()
    z = (np.sin(phi) * np.expand_dims(np.ones((n + 1)), axis=1)).flatten()
    coords = np.stack((x, y, z), -1)
    return coords


def show_point_cloud(cloud, subsample=3):
    """
    Plots the pointcloud as defined by a dictionary containing the keys 'Positions' and 'Colors' using matplotlib scatter.
    Not particularly quick or enjoyable.
    
    Optional functionality to reduce size of the pointcloud by subsampling points. This speeds up the process but decimates the pointcloud.
    
    Input: 
            cloud - A dictionary representing the pointcloud, with keys 'Positions' and 'Colors'.
                  - 'Positions' is an Nx3 numpy array of XYZ locations for each point.
                  - 'Colors' is an Nx3 element numpy array of colours for each point.
    
    Optional: 
            subsample - Scalar value for the number datapoints to skip when plotting the cloud. Default 3.
    """
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(xs=cloud['Positions'][::subsample, 0], 
                ys=cloud['Positions'][::subsample, 1], 
                zs=cloud['Positions'][::subsample, 2], 
                s=20,
                marker='s',
                c=cloud['Colors'][::subsample, :]/255.0)
    plt.title('Global Pointcloud')
    plt.show(block=False)
    
    
def angle_to_directional_cosine(angle):
    """
    Converts a set of yaw, pitch, roll rotations in radians to the direction cosine matrix.
    
    Input: 
            angle - A 3 element numpy vector containing the radian angle rotations in ZYX application order.
    
    Output: 
            C - 3x3 Cosine matrix 
    """
    
    R3 = np.zeros((3, 3))
    R2 = np.zeros((3, 3))
    R1 = np.zeros((3, 3))

    R3[2, 2] = 1.0
    R3[0, 0] = np.cos(angle[0])
    R3[0, 1] = np.sin(angle[0])
    R3[1, 0] = -np.sin(angle[0])
    R3[1, 1] = np.cos(angle[0])

    R2[1, 1] = 1.0
    R2[0, 0] = np.cos(angle[1])
    R2[0, 2] = -np.sin(angle[1])
    R2[2, 0] = np.sin(angle[1])
    R2[2, 2] = np.cos(angle[1])

    R1[0, 0] = 1.0
    R1[1, 1] = np.cos(angle[2])
    R1[1, 2] = np.sin(angle[2])
    R1[2, 1] = -np.sin(angle[2])
    R1[2, 2] = np.cos(angle[2])

    try:
        C = np.einsum('ij, jk, km -> im', R1, R2, R3)
    except AttributeError:
        C = R1.dot(R2.dot(R3))

    return C


def points_to_image(cloud_positions, cloud_colors, image_size, cam, projection):
    """
    Project the current view into the pointcloud to the image plane to produce an image.
    
    Input: 
           cloud_positions - Nx3 numpy array of XYZ locations for each point, i.e. 'Positions' key within a pointcloud dictionary.
           cloud_colors - Nx3 element numpy array of colours for each point, i.e. 'Colors' key within a pointcloud dictionary.
           image_size - 2 element vector of requested image height and width.
           cam - Scalar value for camera focal length
           projection - 4x4 projection matrix for the global system to the image plane.
           
    Output: 
            image - image_size[0] x image_size[1] x 3 RGB image of the pointcloud colours projected to the image plane.
            mapx  - image_size[0] x image_size[1] map of pixel coordinate to X axis value in the global coordinate space.
            mapy  - image_size[0] x image_size[1] map of pixel coordinate to Y axis value in the global coordinate space.
            mapz  - image_size[0] x image_size[1] map of pixel coordinate to Z axis value in the global coordinate space.
    """
    
    disk_filter = np.asarray([[0.025078581023833, 0.145343947430219, 0.025078581023833],
                              [0.145343947430219, 0.318309886183791, 0.145343947430219],
                              [0.025078581023833, 0.145343947430219, 0.025078581023833]])
    disk_filter = 0.5 * disk_filter / np.max(disk_filter)
    cam = np.asarray([[cam, 0, image_size[1] / 2, 0],
           [0, cam, image_size[0] / 2, 0],
           [0, 0, 1, 0]], dtype=np.float64)  
    
    locations = (projection @ np.concatenate([cloud_positions, np.ones((cloud_positions.shape[0], 1))], axis=1).T).T
    dist = np.sum(locations[:, 0:3] ** 2, axis=1)
    idx = np.argsort(-dist, kind='mergesort')   
    locations = locations[idx,:]
    colors = cloud_colors[idx] / 255
    
    locations = (cam @ locations.T).T
    keep = np.where(locations[:, -1] > 0)[0]
    keepidx = idx[keep]
    locations = locations[keep]
    colors = colors[keep]
    locations = locations[:, 0:2] / np.tile(locations[:, 2:3], (1,2))
    locations = np.round(locations).astype(np.int)
    
    keep = np.where((locations[:,0] >= -disk_filter.shape[0]) * \
           (locations[:,0] < (image_size[1] + disk_filter.shape[0])) * \
           (locations[:,1] >= -disk_filter.shape[1]) * \
           (locations[:,1] < (image_size[0] + disk_filter.shape[1])))[0]      
    locations = locations[keep, :]
    colors = colors[keep, :]
    keepidx = keepidx[keep]
    
    image = np.zeros(image_size + [3])
    mapall = np.zeros(image_size + [3])
    for i_point in range(locations.shape[0]):
        for iy in [-1,0,1]:
            lociy = locations[i_point,1]+iy
            if (lociy >= 0) and (lociy < image_size[0]):
                for ix in [-1,0,1]:
                    locix = locations[i_point,0]+ix

                    valid_location = (locix >= 0) and \
                                     (locix < image_size[1])

                    if valid_location:
                        opac = disk_filter[iy+1, ix+1]
                        col = (1-opac) * image[lociy, locix, :]
                        image[lociy, locix, :] = col.flatten() + opac * colors[i_point, :]
                        mapall[lociy, locix, :] = cloud_positions[keepidx[i_point],:]

    mapx = mapall[:,:,0]
    mapy = mapall[:,:,1]
    mapz = mapall[:,:,2]
    return image, mapx, mapy, mapz


def project_pointcloud_image(cloud, angle, position):
    """
    Project a viewpoint into pointcloud and return captured image. Also returns the XYZ maps and the real-world depth map for the image.
    
    Input: 
            cloud - A dictionary with keys 'Positions' and 'Colors'.
                 - 'Positions' is an Nx3 numpy array of XYZ locations for each point.
                 - 'Colors' is an Nx3 element numpy array of colours for each point.
            angle - 3 element numpy vector corresponding to the radian angle rotations of the camera viewpoint in ZYX ordering.
            position - 3 element numpy vector corresponding to the XYZ placement of the camera in world coordinates.
    
    Output:
            image - image_size[0] x image_size[1] x 3 RGB image of the pointcloud colours projected to the image plane.
            mapx  - image_size[0] x image_size[1] map of pixel coordinate to X axis value in the global coordinate space.
            mapy  - image_size[0] x image_size[1] map of pixel coordinate to Y axis value in the global coordinate space.
            mapz  - image_size[0] x image_size[1] map of pixel coordinate to Z axis value in the global coordinate space.
            depth  - image_size[0] x image_size[1] map of pixel coordinate to real-world distance.
    """
    
    # Get camera orientation info
    translation = np.eye(4)
    translation[0:3, 3] = -1 * position
    rotation = np.eye(4)
    rotation[0:3, 0:3] = angle_to_directional_cosine(angle)
    projection = rotation @ translation
    
    # Project points to image plane
    colors = cloud['Colors']
    cam = 300
    image_size = [160,240]
    image, mapx, mapy, mapz = points_to_image(cloud['Positions'], colors, image_size, cam, projection)
    
    dist = np.sqrt((mapx.flatten() - position[0])**2 +
                   (mapy.flatten() - position[1])**2 + 
                   (mapz.flatten() - position[2])**2)
    
    depth = np.reshape(dist, image_size)
    return image, mapx, mapy, mapz, depth
    

def startup_scene(subsample=1):
    """
    Initialise the game pointcloud, populating with all spheres.
    
    Input:
            None
            
    Optional: 
            subsample - Scalar value for the number datapoints to skip when initialising the cloud. Default 1.
            
    Output:
            global_cloud - A dictionary with keys 'Positions' and 'Colors'.
                 - 'Positions' is an Nx3 numpy array of XYZ locations for each point.
                 - 'Colors' is an Nx3 element numpy array of colours for each point.
            spheres_collected - An M element boolean list indicating which of the M spheres have been collected.
    """
    
    # Define sphere locations
    sphere_positions = np.asarray([[-0.1971,0.0620,2.4200],
                                   [-0.3208,-0.0384,4.7844],
                                   [-0.9484,0.2093,7.1190],   
                                   [-1.0448,0.6402,9.4877],   
                                   [-1.9173,0.7783,12.2852],   
                                   [-3.8317,1.0877,12.9989],   
                                   [-6.6664,1.4695,13.2349],   
                                   [-8.9885,1.6683,11.4675],   
                                   [-9.4013,1.8192,9.6671],   
                                   [-9.2761,1.8705,7.1552],
                                   [-9.0310,2.0825,4.4461]])

    # Initialise the score, creating a list of uncollected spheres
    spheres_collected = [False for sphere in range(len(sphere_positions))]
    
    # Load pointcloud data, positions and colors, from numpy files
    #global_cloud = {}
    #global_cloud['Positions'] = np.load('cloudPositions.npy')[0::subsample, :]
    #global_cloud['Colors'] = np.load('cloudColors.npy')[0::subsample, :]
    global_cloud['Positions'] = global_cloud['Positions'][0::subsample, :]
    global_cloud['Colors'] = global_cloud['Colors'][0::subsample, :]
    
    # Create spheres and place them into the pointcloud
    generic_sphere = sphere(200)
    sphere_size = 9    
    for i_sphere in range(len(sphere_positions)):
        sphere_coords = generic_sphere / sphere_size + sphere_positions[i_sphere, :]
        sphere_color = np.tile(np.asarray([255, 0, 0], dtype=np.float64), [sphere_coords.shape[0], 1]);
        global_cloud['Positions'] = np.concatenate([global_cloud['Positions'], sphere_coords])
        global_cloud['Colors'] = np.concatenate([global_cloud['Colors'], sphere_color])
        
    return global_cloud, spheres_collected
    
    
def update_scene(position, spheres_collected, subsample=1):
    """
    Update the game pointcloud, removing spheres that have already been captured. 
    Calculates if the current position is close enough to capture a new sphere.
    
    Input:
            position - 3 element numpy vector corresponding to the XYZ placement of the camera in world coordinates.
            spheres_collected - An M element boolean list indicating which of the M spheres have been collected.
            
    Optional: 
            subsample - Scalar value for the number datapoints to skip when initialising the cloud. Default 1.
            
    Output:
            global_cloud - An updated dictionary with keys 'Positions' and 'Colors'.
                 - 'Positions' is an Nx3 numpy array of XYZ locations for each point.
                 - 'Colors' is an Nx3 element numpy array of colours for each point.
            spheres_collected - An updated M element boolean list indicating which of the M spheres have been collected.
    """
    
    # Define sphere locations
    sphere_positions = np.asarray([[ -0.1971,0.0620,2.4200],
                                   [-0.3208,-0.0384,4.7844],
                                   [-0.9484,0.2093,7.1190],   
                                   [-1.0448,0.6402,9.4877],   
                                   [-1.9173,0.7783,12.2852],   
                                   [-3.8317,1.0877,12.9989],   
                                   [-6.6664,1.4695,13.2349],   
                                   [-8.9885,1.6683,11.4675],   
                                   [-9.4013,1.8192,9.6671],   
                                   [-9.2761,1.8705,7.1552],
                                   [-9.0310,2.0825,4.4461]]) 
    
    # Update the score, updating spheres_collected and removing the sphere from the cloud
    sphere_size = 9
    dist = cdist(np.expand_dims(position, axis=0), sphere_positions)
    dist_threshold = 1.0 / (sphere_size - 1)
    points_to_remove = np.where(dist <= dist_threshold)[1][0]
    spheres_collected[points_to_remove] = True
    spheres_to_render = [not elem for elem in spheres_collected]
    sphere_positions = sphere_positions[spheres_to_render, :]
        
    # Load pointcloud data, positions and colors, from numpy files
    global_cloud = {}
    global_cloud['Positions'] = np.load('cloudPositions.npy')[0::subsample, :]
    global_cloud['Colors'] = np.load('cloudColors.npy')[0::subsample, :]
    #global_cloud['Positions'] = global_cloud['Positions'][0::subsample, :]
    #global_cloud['Colors'] = global_cloud['Colors'][0::subsample, :]

    # Create remaining spheres and place them into the pointcloud
    generic_sphere = sphere(200)
    for i_sphere in range(len(sphere_positions)):
        sphere_coords = generic_sphere / sphere_size + sphere_positions[i_sphere, :]
        sphere_color = np.tile(np.asarray([255, 0, 0], dtype=np.float64), [sphere_coords.shape[0], 1]);
        global_cloud['Positions'] = np.concatenate([global_cloud['Positions'], sphere_coords])
        global_cloud['Colors'] = np.concatenate([global_cloud['Colors'], sphere_color]) 
        
    return global_cloud, spheres_collected