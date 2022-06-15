import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class TIN:
    """
    Class representing a TIN (Triangulated Irregular Network)

    Atributes:
    ------------
    coords: coordinates on the x and y axis, stored in a numpy array
    triang: Delaunay triangulation
    elevs: coordinates on the z axis, stored in a numpy array

    Public methods:
    -------------
    plot3d(): plots the triangulated data in 3D.
    plot2d(anotate = False): plots the triangulated data in 2D, and the elevation of each point if indicated.
        This last thing is not recommended if you have many points close together. By convention, from lowest 
        to highest altitude it is red, orange and green.
    point_elevation(pt, plot = False): interpolates the elevation of the given point, and plots if indicated.
    largest_drainage(pt, plot = False): finds the largest-area drainage basin of the given point, and plots
        if indicated.
    triang_quality(plot = False): reports the minimum and maximum angle of the triangulation, and plots if
        indicated.
    complete_pipe_network(plot = False): finds the minimal Euclidean spanning tree of the triangulation, and
        plots if indicated.
    
    """

    def __init__(self,pts):
        """
        Parameters:
        -------------
        pts: a numpy array of dimension (N,3), where N is the total of vertices

        """
        self.coords = np.delete(pts,2,axis=1)
        self.triang = Delaunay(self.coords)
        self.elevs = pts[:,2]

    def __create_colormap(self):
        """
        Create three categories based on the elevation of each point

        Output
        ----------
        Numpy array with the categories
        
        """
        ranges = [min(self.elevs) + (k)*(max(self.elevs)/3) for k in range(4)]
        cat = [0]*len(self.elevs)
        for i in range(3):
            cat = np.where((ranges[i] <= self.elevs) & (self.elevs <= ranges[i+1]),i,cat)
        return np.array(cat)

    def plot3d(self):
        """
        Plots the triangulated data in 3D
        """
        tri = mtri.Triangulation(x=self.coords[:, 0], y=self.coords[:, 1], triangles=self.triang.simplices)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        g = ax.plot_trisurf(tri, self.elevs, cmap = "plasma")
        cbar = fig.colorbar(g)
        plt.title("Triangulation of data in 3D")
        plt.show()

    def plot2d(self, annotate = False):
        """
        Plots the triangulated data in 2D

        Parameters
        -------------
        annotate: boolean, if True, write down its altitude next to each vertex
        """
        fig, ax = plt.subplots()
        categories = self.__create_colormap()
        colormap = np.array(["red", "orange", "green"])
        plt.triplot(self.coords[:, 0], self.coords[:, 1], self.triang.simplices, zorder = 1)
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c=colormap[categories], zorder = 2)
        if annotate:
            for i, elev in enumerate(self.elevs):
                ax.annotate(round(elev,3), (self.coords[i, 0], self.coords[i, 1]), weight='bold', fontsize=10)
        plt.title("Triangulation of data in 2D")
        plt.show()

    def __find_triangle(self, pt):
        """
        Find the triangle in which the given point is located
        
        Parameters
        --------------
        pt: point stored in a numpy array

        Output
        --------------
        t_points: numpy array with the coordinates of the vertices of the triangle
        e_points: numpy array with the elevations of the vertices of the triangle
        triangle_index: index of the triangle within the list of simplest
        """
        triangle_index = self.triang.find_simplex(pt)
        triangle = self.triang.simplices[triangle_index]

        t_points = np.array([self.coords[triangle[0]], self.coords[triangle[1]], self.coords[triangle[2]]])
        e_points = np.array([self.elevs[triangle[0]], self.elevs[triangle[1]], self.elevs[triangle[2]]])

        return t_points, e_points, triangle_index

    def point_elevation(self, pt, plot = False):
        """
        Computes the elevation of the given point using piecewise linear interpolation

        Parameters
        ---------------
        pt: point stored in a numpy array
        plot: boolean, if True, plots the given point in blue, and the vertices of the triangle that store it in red

        Output
        ---------------
        elev_pt: elevation of the point

        """
        t_points, e_points, triang_index = self.__find_triangle(pt)

        if triang_index == -1:
            print("The entered point is not in the valid ranges of the triangulation coordinates.")
            return None

        interpolator = LinearNDInterpolator(t_points, e_points)
        elev_pt = interpolator(pt)[0]

        if plot:
            tri = mtri.Triangulation(x=self.coords[:, 0], y=self.coords[:, 1], triangles=self.triang.simplices)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            g = ax.plot_trisurf(tri, self.elevs, alpha = 0.1, color="grey", edgecolor="grey")

            ax.scatter(t_points[0][0], t_points[0][1], e_points[0], 'o', c='red')
            ax.scatter(t_points[1][0], t_points[1][1], e_points[1], 'o', c='red')
            ax.scatter(t_points[2][0], t_points[2][1], e_points[2], 'o', c='red')
            ax.scatter(pt[0],pt[1],elev_pt, c="blue")

            plt.title("Interpolation of the elevation of a point")
            plt.show()

        return(elev_pt)

    def __find_angle(self,A,B,C):
        """
        Calculate the angle between vectors AB and AC
        
        Parameters
        ------------
        A: coordinates of point A stored in a numpy array, can be in 2 or 3 dimensions
        B: coordinates of point B stored in a numpy array, can be in 2 or 3 dimensions
        C: coordinates of point C stored in a numpy array, can be in 2 or 3 dimensions

        Output
        ------------
        Returns the angle in degrees

        """
        vec_1 = (B-A)
        vec_2 = (C-A)
        rad_angle = np.arccos(np.dot(vec_1,vec_2)/( np.linalg.norm(vec_1)*np.linalg.norm(vec_2)))
        return rad_angle*180/np.pi

    def __triangle_area(self,A,B,C,theta):
        """
        Calculate the area of the triangle given its three points and the angle between the vectors AB and AC

        Parameters
        --------------
        A: coordinates of point A stored in a numpy array, can be in 2 or 3 dimensions
        B: coordinates of point B stored in a numpy array, can be in 2 or 3 dimensions
        C: coordinates of point C stored in a numpy array, can be in 2 or 3 dimensions
        theta: angle between AB and AC

        Output
        --------------
        Returns the area of the given triangle
        
        """
        return (0.5)*np.linalg.norm(B-A)*np.linalg.norm(C-A)*np.sin(theta)

    def largest_drainage(self,pt,plot = False):
        """
        Finds the largest-area drainage basin of a given point

        Parameters
        -------------
        pt: point stored in a numpy array
        plot: boolean, if True, plots the given point in blue, and the vertices of the drainage found in red

        Output
        -------------
        largest_d: numpy array containing the vertices of the largest-area draining basin
        max_area: area of the drainage
        
        """
        if pt.tolist() not in self.coords.tolist():
            t_points, e_points, triangle_index = self.__find_triangle(pt)

            if triangle_index == -1:
                print("The entered point is not in the valid ranges of the triangulation coordinates.")
                return None, None

            points_3d = np.array([np.append(t_points[0], e_points[0]), np.append(t_points[1], e_points[1]), np.append(t_points[2], e_points[2])])

            area_t = self.__triangle_area(points_3d[0], points_3d[1], points_3d[2], self.__find_angle(points_3d[0], points_3d[1], points_3d[2])*np.pi/180)

            neighbors_index = self.triang.neighbors[triangle_index]
            neighbors = self.triang.simplices[neighbors_index]

            quad_areas = []
            for neigh in neighbors:
                a = np.append(self.coords[neigh[0]],self.elevs[neigh[0]])
                b = np.append(self.coords[neigh[1]],self.elevs[neigh[1]])
                c = np.append(self.coords[neigh[2]],self.elevs[neigh[2]])
                quad_areas.append(area_t + self.__triangle_area(a,b,c,self.__find_angle(a,b,c)*np.pi/180))

            max_index = np.argmax(quad_areas)
            max_area = max(quad_areas)
            max_quad_index = neighbors[max_index]

            max_quad = np.append(self.coords[max_quad_index],self.elevs[max_quad_index].reshape(-1,1),axis=1)
            largest_d = np.unique(np.concatenate((points_3d, max_quad), axis=0), axis=0)
            
        else:
            coords = self.coords.tolist()
            position_pt = coords.index(pt.tolist())

            incident_triangles_indexes = []
            for i, tri in enumerate(self.triang.simplices):
                if position_pt in tri.tolist():
                    incident_triangles_indexes.append(i)

            possible_pairs = []
            for i in incident_triangles_indexes:
                neighbors_index = self.triang.neighbors[i]
                for j in neighbors_index:
                    pair = set([i,j])
                    if pair not in possible_pairs:
                        possible_pairs.append(pair)
                        
            possible_areas = []
            for pair in possible_pairs:
                T1 = self.triang.simplices[list(pair)[0]]
                T2 = self.triang.simplices[list(pair)[1]]
                a1 = np.append(self.coords[T1[0]], self.elevs[T1[0]])
                b1 = np.append(self.coords[T1[1]], self.elevs[T1[1]])
                c1 = np.append(self.coords[T1[2]], self.elevs[T1[2]])
                a2 = np.append(self.coords[T2[0]], self.elevs[T2[0]])
                b2 = np.append(self.coords[T2[1]], self.elevs[T2[1]])
                c2 = np.append(self.coords[T2[2]], self.elevs[T2[2]])
                possible_areas.append(self.__triangle_area(a1,b1,c1,self.__find_angle(a1,b1,c1)*np.pi/180) + self.__triangle_area(a2,b2,c2,self.__find_angle(a2,b2,c2)*np.pi/180))
            
            max_index = np.argmax(possible_areas)
            max_area = max(possible_areas)
            max_pair = list(possible_pairs[max_index])
            T1_max = self.triang.simplices[max_pair[0]]
            T2_max = self.triang.simplices[max_pair[1]]
            points_T1 = []
            points_T2 = []

            for i in range(3):
                p = np.append(self.coords[T1_max[i]], self.elevs[T1_max[i]])
                q = np.append(self.coords[T2_max[i]], self.elevs[T2_max[i]])
                points_T1.append(p)
                points_T2.append(q)

            largest_d = np.unique(np.concatenate((points_T1, points_T2), axis=0), axis=0)

        if plot: 
                tri = mtri.Triangulation(x=self.coords[:, 0], y=self.coords[:, 1], triangles=self.triang.simplices)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                g = ax.plot_trisurf(tri, self.elevs, alpha=0.1, color="grey", edgecolor="grey")

                ax.scatter(largest_d[0][0], largest_d[0][1], largest_d[0][2], 'o', c = 'red')
                ax.scatter(largest_d[1][0], largest_d[1][1], largest_d[1][2], 'o', c = 'red')
                ax.scatter(largest_d[2][0], largest_d[2][1], largest_d[2][2], 'o', c = 'red')
                ax.scatter(largest_d[3][0], largest_d[3][1], largest_d[3][2], 'o', c='red')
                ax.scatter(pt[0], pt[1], self.point_elevation(pt,False), c="blue")

                plt.title("Largest-area draining basin of given point")
                plt.show()
        
        return largest_d, max_area


    def triang_quality(self, plot = False):
        """
        Measures the quality of the triangulation by calculating its minimum and maximum angle

        Parameters
        --------------
        plot: boolean, if True, graph the triangulation, indicating the vertex with the greatest angle in red 
            and the one with the smallest angle in green, as well as their values. If both correspond to the 
            same vertex, it is indicated in yellow.

        Output
        --------------
        A tuple with the minimum and maximum angle in degrees
        
        """
        for j, tri in enumerate(self.triang.simplices):
            local_angles = []
            points = [self.coords[tri[0]], self.coords[tri[1]], self.coords[tri[2]]]

            for i in range(3):
                local_angles.append(self.__find_angle(points[i], points[(i+1) % 3], points[(i+2) % 3]))

            if j == 0:
                min_angle = [min(local_angles),j,np.argmin(local_angles)]
                max_angle = [max(local_angles),j,np.argmax(local_angles)]
            else:
                if min(local_angles) <= min_angle[0]:
                    min_angle = [min(local_angles), j, np.argmin(local_angles)]
                if max(local_angles) >= max_angle[0]:
                    max_angle = [max(local_angles), j, np.argmax(local_angles)]

        if plot:
            fig, ax = plt.subplots()
            plt.triplot(self.coords[:, 0],self.coords[:, 1], self.triang.simplices, zorder = 1)
            plt.scatter(self.coords[:, 0], self.coords[:, 1], zorder=1)

            triang_min = self.triang.simplices[min_angle[1]]
            points_min = triang_min[min_angle[2]]
            triang_max = self.triang.simplices[max_angle[1]]
            points_max = triang_max[max_angle[2]]

            if (self.coords[points_min][0] != self.coords[points_max][0]) and (self.coords[points_min][1] != self.coords[points_max][1]):
                plt.scatter(self.coords[points_min][0], self.coords[points_min][1], c="g", zorder = 2)
                ax.annotate(round(min_angle[0],3), (self.coords[points_min][0],self.coords[points_min][1]), weight='bold', fontsize=10)
                plt.scatter(self.coords[points_max][0], self.coords[points_max][1], c="r", zorder = 2)
                ax.annotate(round(max_angle[0],3), (self.coords[points_max][0],self.coords[points_max][1]), weight='bold', fontsize=10)

            else: 
                plt.scatter(self.coords[points_min][0], self.coords[points_min][1], c="y", zorder=2)
                ax.annotate(str(round(min_angle[0],3)) + ", " + str(round(max_angle[0],3)), (self.coords[points_max]
                                                                           [0], self.coords[points_max][1]), weight='bold', fontsize=10)
            plt.title("Measuring the triangulation's quality")
            plt.show()

        return min_angle[0], max_angle[0]


    def complete_pipe_network(self,plot = False):
        """
        Computes the complete pipe network using a Euclidean minimal spanning tree

        Parameters
        ------------
        plot: boolean, if True, plots the spanning tree

        Output
        ------------
        Numpy array with spanning tree adjacency matrix
        
        """
        G = np.zeros((len(self.coords), len(self.coords)))

        for i in range(len(self.coords)):
            vertex_neighbors = self.triang.vertex_neighbor_vertices[1][self.triang.vertex_neighbor_vertices[0][i]:self.triang.vertex_neighbor_vertices[0][i+1]]
            initial_point = self.coords[i]

            for j in vertex_neighbors:
                final_point = self.coords[j]
                distance = np.linalg.norm(final_point - initial_point)
                G[i,j] = distance 
                G[j,i] = distance 

        X = csr_matrix(G)
        Tcsr = minimum_spanning_tree(X)

        if plot:
            non_zero_entries = np.nonzero(Tcsr.toarray())
            fig, ax = plt.subplots()
            plt.triplot(self.coords[:, 0],self.coords[:, 1], self.triang.simplices, alpha = 0.2, color = "grey")
            plt.scatter(self.coords[:, 0], self.coords[:, 1], alpha = 0.3, color = "black", s = 1)

            for i in range(len(non_zero_entries[0])):
                point_1 = self.coords[non_zero_entries[0][i]]
                point_2 = self.coords[non_zero_entries[1][i]]
                plt.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], "ro-", linewidth=1, markersize=3)

            plt.title("Complete pipe network")
            plt.show()
        return Tcsr.toarray()

"""
Small test with some data.
"""

if __name__ == "__main__":


    np.random.seed(0)
    #Generate random coords and elevations
    coords = 3*np.random.random_sample((20,2))
    elevs = 3*np.random.random_sample((20,1))
    data = np.append(coords,elevs.reshape(-1,1),axis=1)

    # Initializing our TIN
    tin = TIN(data)

    # Visualizing the triangulated data in 3D
    tin.plot3d()

    # Visualizing the triangulated data in 2D
    tin.plot2d(True)

    # Let us generate a point
    x_upper_limit, x_lower_limit = np.max(coords[:, 0]), np.min(coords[:, 0])
    y_upper_limit, y_lower_limit = np.max(coords[:, 1]), np.min(coords[:, 1])

    x = np.random.uniform(x_lower_limit, x_upper_limit)
    y = np.random.uniform(y_lower_limit, y_upper_limit)
    p = np.array([x, y])
    # And calculate its elevation

    p_elevation = tin.point_elevation(p,True)
    if p_elevation != None:
        print("Elevation at point",p,":",round(p_elevation,5))

    # Let us also find its largest area-drainage basin
    largest_drainage_basin, area = tin.largest_drainage(p,True)
    if area != None:
        print("Largest drainage basin at",p,":",largest_drainage_basin,"\n","Its area is ",round(area,5))

    # Measuring the triangulation's quality...
    min_angle, max_angle = tin.triang_quality(True)
    print("The minimum angle is ",round(min_angle,5),", and the maximum is ",round(max_angle,5))

    # Finallye, let us calculate the complete pipe network
    tin.complete_pipe_network(True)
