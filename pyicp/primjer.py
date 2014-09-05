import pyicp
from pyicp import calculate_normals

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from itertools import islice
import random


def show_normals(ys, num_neighbors=10):
    tree = cKDTree(ys)
    ns = calculate_normals(tree, num_neighbors)
    plt.quiver(ys[:,0], ys[:,1], ns[:,0], ns[:,1], angles='xy')

    plt.plot(ys[:,0], ys[:,1], 'o')
    plt.show()


def example1():
    t = np.linspace(-10, 10, 100)

    ys = np.array([t, 0.1 * t * t]).T
    show_normals(ys, 10)

    ys = np.array([t, 5 * np.sin(t)]).T
    show_normals(ys, 5)


def example2():
    phi = -np.pi / 24
    px = 0.5
    py = 0.5

    R, p = pyicp.get_matrices([phi, px, py])

    t = np.linspace(-10, 10, 1000)
    #source_points = np.array([t, 0.1 * t * t]).T
    source_points = np.array([t, np.sin(t)]).T
    #source_points = np.random.multivariate_normal([0,0],[[100, 0],[0,1]], 100)

    target_points = np.dot(source_points, R.T) + p

    R, p, j = pyicp.find_transformation(source_points, target_points,
                                        num_neighbors=10, num_iterations=20,
                                        indist=0.05)

    tranformed_points = np.dot(source_points, R.T) + p
    print "criterion =", j

    plt.plot(target_points[:,0], target_points[:,1], 'bo')
    plt.plot(source_points[:,0], source_points[:,1], 'ro')
    plt.plot(tranformed_points[:,0], tranformed_points[:,1], 'go')

    plt.show()


if __name__ == '__main__':
    example2()
