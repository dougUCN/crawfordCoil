from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.spatial import ConvexHull
np.seterr(divide='ignore', invalid='ignore')

# Returns intersection of a ray U (starting from 0) with a convex hull
# https://stackoverflow.com/questions/30486312/intersection-of-nd-line-with-convex-hull-in-python
def hit(U,hull):
    eq=hull.equations.T
    V,b=eq[:-1],eq[-1]
    alpha=-b/np.dot(U,V)
    return np.min(alpha[alpha>0])*U

# Intersection of multiple rays U with a convex hull
def manyHit(U,hull):
    hitpoints = []
    for Uvec in U:
        hitpoints.append(hit(Uvec,hull))
    return np.array(hitpoints)

# Converts to spherical coordinates
def toSpherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(np.array(z)/r)
    return r, theta, phi

# Converts to cartesian coordinates
def toCart(rad, theta, phi):
    x = rad * np.sin(theta) * np.cos(phi)
    y = rad * np.sin(theta) * np.sin(phi)
    z = rad * np.cos(theta)
    return x, y, z

# Generates a spherical hull
def sphereHull(rad=1, thetaMin=0, thetaMax=np.pi, phiMin=0, phiMax=2*np.pi, numTheta=1E3, numPhi=1E3):
    phi = np.random.uniform(phiMin, phiMax, int(numPhi))
    theta = np.random.uniform(thetaMin, thetaMax, int(numTheta))
    x, y , z = toCart(rad, theta, phi)
    return ConvexHull( np.array([x,y,z]).T )

# Generates currentLoops from contours
# countours is a matplotlib.countour object in theta-phi space
# hull is the ConvexHull geometry onto which the current loops will be projected
# sphereR is used to convert from theta-phi space to cartesian space
#
# If thetaBounds [thetaMin, thetaMax] and phiBounds [phiMin, phiMax] are specified,
# then current loop coordinates outside of these values will be removed
#
# Returns:
# currentLoops = [loop0, loop1, loop2, ...]
# loop0 = [[x0,y0,z0], [x1,y1,z1], ...]
# Where loops have been projected back into cartesian coords onto the hull
def genLoops(contours, hull, thetaBounds=None, phiBounds=None, sphereR=1):
    # From documentation on matplotlib.contour:
    # allsegs : [level0segs, level1segs, ...]
    # level0segs = [polygon0, polygon1, ...]
    # polygon0 = array_like [[x0,y0], [x1,y1], ...]
    currentLoops = []
    for levelsegs in contours.allsegs:
        for polygon in levelsegs:
            tempLoop = []
            for xy in polygon:
                # xy[0] = theta, xy[1] = phi
                if thetaBounds and phiBounds and (xy[0]<thetaBounds[0] or xy[0]>thetaBounds[1] \
                                                    or xy[1]<phiBounds[0] or xy[1]>phiBounds[1]):
                    continue
                i, j, k= toCart(sphereR, xy[0], xy[1])
                tempLoop.append(i)
                tempLoop.append(j)
                tempLoop.append(k)
            if not tempLoop:
                # Empty loops should be ignored
                continue
            tempLoop = np.reshape(tempLoop, (-1,3) )
            # project each loop back to cylindrical hull
            currentLoops.append( manyHit(tempLoop, hull) )
    return currentLoops

# Given values on a sphere theta=[0,pi]; phi=[-pi, pi]; with scalar potential A;
# makes the boundaries semi-periodic by appending additional values such that
# theta=[-bw, pi+bw]; phi=[-(pi + bw), pi+bw], where bw is the boundaryWidth
def semiPeriodicBounds(theta, phi, A, boundaryWidth):
    thetaMin = np.amin(theta)
    thetaMax = np.amax(theta)
    thetaEdge1 = np.all([theta>(thetaMax-boundaryWidth), theta<thetaMax], axis=0, keepdims=True)[0]
    thetaEdge2 = np.all([theta<(thetaMin+boundaryWidth), theta>thetaMin], axis=0, keepdims=True)[0]
    phiMin = np.amin(phi)
    phiMax = np.amax(phi)
    phiEdge1  = np.all([phi>(phiMax-boundaryWidth), phi<phiMax] , axis=0, keepdims=True)[0]
    phiEdge2 = np.all([phi<(phiMin+boundaryWidth), phi>phiMin], axis=0, keepdims=True)[0]

    theta = np.concatenate((np.array(theta), theta[thetaEdge1] - np.pi, theta[thetaEdge2] + np.pi, theta[phiEdge1], theta[phiEdge2]), axis=None)
    phi = np.concatenate((np.array(phi), phi[thetaEdge1], phi[thetaEdge2], phi[phiEdge1] - 2*np.pi, phi[phiEdge2] + 2*np.pi ), axis=None)
    A = np.concatenate((A, A[thetaEdge1], A[thetaEdge2], A[phiEdge1], A[phiEdge2]), axis=None)
    return theta, phi, A

# Simple conditional to remove mesh triangles according to x and y bounds
# xBounds = [x0, x1]; yBounds = [y0,y1]
# Mesh should be a tri.Triangulation object, and x y should be corresponding
# coordinate values
def maskBounds(x, xBounds, y, yBounds, mesh):
    # print(mesh.triangles.shape[0])    # number of triangles in mesh
    xMid = np.array( x[mesh.triangles].mean(axis=1) ) #Finds midpoint of triangles
    yMid = np.array( y[mesh.triangles].mean(axis=1) )
    mask = np.any( [xMid<xBounds[0] , xMid>xBounds[1], yMid<yBounds[0], yMid>yBounds[1]], axis=0)
    mesh.set_mask(mask)
    print(mesh.triangles.shape[0])
    return mesh


# Assigns points on meshes to their closest guard
# def voronoi():

# Export to text file
# def export():
