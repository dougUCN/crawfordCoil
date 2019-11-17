#!/usr/bin/env python
# Draws equipotential lines on a cylinder, using multiple "guard" points
# First, points on the cylinder are divided according their nearest "guard"
# Then, for each guard, the method implemented in drawCyl.py is carried out
# While this example utilizes a cylinder, for more complicated geometries
# this method is required

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.spatial import ConvexHull
import crawfordCoilLib as cc

def main():
    xGuard, yGuard, zGuard = np.genfromtxt("guards.txt", comments='%', unpack=True)
    xVals, yVals, zVals, aVals = np.genfromtxt("cylinder-surface.txt", comments='%', unpack=True)

    cylPoints, cylA = cc.assignPoints(xVals, yVals, zVals, aVals, xGuard, yGuard, zGuard)

    # Generate current loops for each set of guard points
    allLoops=[]
    loopColor=[]
    fig, ax = plt.subplots()
    for i, (set, A) in enumerate(zip(cylPoints,cylA)):
        # Shift points so that the guards are at the origin
        shifted = cc.shiftPoints(set, -xGuard[i], -yGuard[i], -zGuard[i])
        xCyl, yCyl, zCyl = cc.getXYZ(shifted)

        cylHull = cc.genHull(xCyl, yCyl, zCyl)
        rad, theta, phi = cc.toSpherical(xCyl, yCyl, zCyl)
        mesh = tri.Triangulation(theta, phi)
        contours = ax.tricontour(mesh, A, levels=20)
        currentLoops = cc.genLoops(contours, cylHull)
        plt.close()

        # Shift current loops back to correct position
        for loop in currentLoops:
            allLoops.append( cc.shiftPoints(loop, xGuard[i], yGuard[i], zGuard[i]))
            loopColor.append('C' + str(i))

    # Plot current loops
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for loop, color in zip(allLoops, loopColor):
        xLoop, yLoop, zLoop = cc.getXYZ(loop)
        ax.scatter(xLoop, yLoop, zLoop, s=0.3, c=color)

    # Plot points assigned to each guard
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for set in cylPoints:
        xCyl, yCyl, zCyl = cc.getXYZ(set)
        ax.scatter(xCyl,yCyl,zCyl, s=1)

    plt.show()

    return

if ( __name__ == '__main__' ):
    main()
