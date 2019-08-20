# Draws equipotential lines on a cylinder
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.spatial import ConvexHull
import crawfordCoilLib as cc


xCyl, yCyl, zCyl, A = np.genfromtxt("cylinder-surface.txt", comments='%', unpack=True)
cylHull = ConvexHull( np.array( (xCyl.T,yCyl.T,zCyl.T) ).T )

# Convert to theta-phi space
# Add slightly periodic boundaries for nicer contouring
rad, theta, phi = cc.toSpherical(xCyl, yCyl, zCyl)

boundaryWidth = 0.2
thetaMin = np.amin(theta)
thetaMax = np.amax(theta)
phiMin = np.amin(phi)
phiMax = np.amax(phi)

theta, phi, A = cc.semiPeriodicBounds(theta, phi, A, boundaryWidth)

# Triangulate
mesh = tri.Triangulation(theta, phi)
# mesh = cc.maskBounds(theta, [0,np.pi], phi, [-np.pi,np.pi], mesh)

# Draw contour lines
fig, ax = plt.subplots()
contours = ax.tricontour(mesh, A, levels=20)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
ax.set_xlim(thetaMin, thetaMax)
ax.set_ylim(phiMin, phiMax)

# Project contours onto cylinder and generate current loops
# currentLoops = [loop0, loop1, loop2, ...]
# loop0 = [[x0,y0,z0], [x1,y1,z1], ...]
# Since we had semi-periodic boundaries, drop points outside old bounds
currentLoops = cc.genLoops(contours, cylHull, [thetaMin, thetaMax], [phiMin, phiMax])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for loop in currentLoops:
    ax.scatter(loop[:,0], loop[:,1], loop[:,2], s=0.3)

plt.show()
