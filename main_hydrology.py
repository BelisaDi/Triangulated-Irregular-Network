from tin import *
import numpy as np

"""
Read and store a file with data in a numpy array

Parameters
------------
filename: filename, including its extension
sep: data separator

Output
------------
Numpy array with the data

"""
def read_file(filename,sep):
    data = []
    with open(filename, 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.split(sep)
            if k[-1] == "\n":
                k.pop()
            data.append([float(x) for x in k])
    return np.array(data)


np.random.seed(123)

# Reading data
datos = read_file("pts1000c.dat", " ")

# Generating a random point
x_upper_limit, x_lower_limit = np.max(datos[:, 0]), np.min(datos[:, 0])
y_upper_limit, y_lower_limit = np.max(datos[:, 1]), np.min(datos[:, 1])

x = np.random.uniform(x_lower_limit, x_upper_limit)
y = np.random.uniform(y_lower_limit, y_upper_limit)

# We can use the random point, or a point of the sampled
p = np.array([x, y]) 
#p = np.array([datos[200,0], datos[200,1]])

# Initializing our TIN
tin = TIN(datos)

# Visualizing the triangulated data in 3D
tin.plot3d()

# Visualizing the triangulated data in 2D
tin.plot2d()

# Let us consider an arbitrary point, and interpolate its elevation
p_elevation = tin.point_elevation(p,True)
if p_elevation != None:
    print("Elevation at point", p, ":", round(p_elevation, 5))

# Let us also find its largest area-drainage basin
largest_drainage_basin, area = tin.largest_drainage(p,True)
if area != None:
    print("Largest drainage basin at", p, ":", largest_drainage_basin, "\n", "Its area is ", round(area, 5))

# Measuring the triangulation's quality...
min_angle, max_angle = tin.triang_quality(True)
print("The minimum angle is ", round(min_angle, 5), ", and the maximum is ", round(max_angle, 5))

# Finallye, let us calculate the complete pipe network
tin.complete_pipe_network(True)
