import numpy as np
from numpy import dot
from numpy.linalg import svd, lstsq

from scipy.spatial import cKDTree

import functools as fcn
import random


def find_normal(neighbor_points):
    mu = neighbor_points.mean(axis=0)
    dxs = neighbor_points - mu  # zero-centered points

    U, W, V = svd(dxs)

    return V[-1, :]  # vector with least singular value


def calculate_normals(tree, num_neighbors=10):

    _, idxss = tree.query(tree.data, num_neighbors)
    neighborss = [tree.data[idxs] for idxs in idxss]
    normals = np.array(map(find_normal, neighborss))

    return normals


def icp2d_step(target_tree, normals, source_points):

    _, idxss = target_tree.query(source_points)

    ts = np.array([target_tree.data[idxs] for idxs in idxss])
    ns = np.array([normals[idxs] for idxs in idxss])

    ss = source_points

    b = np.array(map(np.dot, ss - ts, ns))

    apxy = -ns
    aphi = np.array(map(np.dot, ns, np.vstack((ss[:, 1], -ss[:, 0])).T))
    aphi.shape = (aphi.size, 1)

    A = np.hstack((aphi, apxy))

    x, j, _, _ = lstsq(A, b)

    return x, j


def get_matrices(x):
    phi, px, py = x
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    p = np.array([px, py])
    return R, p


def find_transformation(source_points, target_points, num_neighbors=10,
                        num_iterations=100):

    dmax = max(source_points.std(), target_points.std())
    ss = source_points / dmax
    ts = target_points / dmax

    target_tree = cKDTree(ts)
    normals = calculate_normals(target_tree, num_neighbors)

    icp_fit = fcn.partial(icp2d_step, target_tree, normals)

    x, j = icp_fit(ss)

    for _ in xrange(num_iterations):
        R, p = get_matrices(x)
        tfs = np.dot(ss, R.T) + p
        dx, j = icp_fit(tfs)
        x += dx

    x[1:] *= dmax

    R, p = get_matrices(x)

    return R, p, j
