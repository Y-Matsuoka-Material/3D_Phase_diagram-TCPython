# coding: UTF-8
import numpy as np

triangle = np.array([(-1, 1 / (3 ** 0.5)), (1, 1 / (3 ** 0.5)), (0, -2 / (3 ** 0.5))])


def gen_mesh(n):
    n = n + 2
    g2 = []
    g3 = []
    for i in range(n):
        current = int((n ** 2 - n) / 2 + i)
        for j in range(i + 1):
            g2.append(current)
            current -= n - j
    for i in range(n):
        current = int((n ** 2 + n - 2) / 2 - i)
        for j in range(i + 1):
            g3.append(current)
            current -= n - j - 1
    g2 = np.array(g2)
    g3 = np.array(g3)
    return g2, g3


def gen_comp(n):
    c1 = []
    c2 = []
    c3 = []
    for i in range(-1, n + 2):
        for j in range(-1, n - i + 1):
            cc1 = i / (n - 1)
            cc2 = j / (n - 1)
            cc3 = (n - 1 - i - j) / (n - 1)
            c1.append(cc1)
            c2.append(cc2)
            c3.append(cc3)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)
    return c1, c2, c3


def gen_verts_raw(func, n):
    c1, c2, c3 = gen_comp(n)

    pot = func(c1, c2, c3)
    pot[(c1 < 0) | (c2 < 0) | (c3 < 0)] = np.nan

    tmp = np.c_[c1, c2, c3] @ triangle
    vertices = np.c_[tmp[:, 0], pot, tmp[:, 1]][::-1]

    g2, g3 = gen_mesh(n)
    vertices_all = np.concatenate((vertices, vertices[g2], vertices[g3]), axis=0)

    return vertices_all


def gen_frame():
    p0, p1, p2, p3, p4, p5 = np.concatenate((np.c_[triangle[:, 0], np.full(3, 1), triangle[:, 1]],
                                             np.c_[triangle[:, 0], np.full(3, -1), triangle[:, 1]]), axis=0)
    vertices_frame = np.array([p0, p1, p1, p2, p2, p0, p3, p4, p4, p5, p5, p3, p0, p3, p1, p4, p2, p5])
    return vertices_frame
