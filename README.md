crawfordCoil
===============

This minimal library uses the 'Chris Crawford method' (University of Kentucky) for  
generating B0 and other guide coils for various geometries.  

The supposedly clever part of this library that makes current loop generation very clean  
is the fact that for any arbitrary geometry, we project the contour map onto theta-phi space (a sphere)  
before drawing the current loops. Then the loops are projected back onto the original geometry

This method works even for geometries that do not project well onto theta phi space  
(i.e. any geometry that would have more than one value at a point on the theta-phi plane)  
This library supports dividing the geometry into well behaved parts, and then stiches the results back together  

Example programs
-----------------
drawCyl.py -- Draws contour lines on a cylinder   (requires cylinder-surface.txt)  

drawCylVoronoi -- Does the same thing as drawCyl.py, except it splits the cylinder  
                  into 3 parts, draws the contour lines on the parts, and then stiches  
                  the current loops back together. The segmenting of the cylinder is  
                  dictated by observer points, or 'guards', (defined in guards.txt).  
