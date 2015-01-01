#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tensors and statistics for joint central moments.

"""
from __future__ import division
import math

def array(size):
    return([0] * size)

def matrix_multiply(a, b):
    # am: # rows in a
    # bm: # rows in b
    # an: # columns in a
    # bn: # columns in b
    am = len(a)
    an = len(a[0])
    bm = len(b)
    bn = len(b[0])
    cm = am
    cn = bn
    c = array(cm)
    if bn > 1:
        i = 0
        while i < cm:
            c[i] = array(cn)
            i += 1
    i = 0
    while i < cm:
        j = 0
        while j < cn:
            k = 0
            while k < an:
                if bn == 1:
                    c[i] += a[i][k] * b[k]
                else:
                    c[i][j] += a[i][k] * b[k][j]
                k += 1
            j += 1
        i += 1
    return(c)

def kron(u, v, size):
    # Calculates the Kronecker product.
    #
    # Args:
    #   u: numeric array (vector)
    #   v: numeric array (vector)
    #   size (int): number of elements in u
    #
    prod = array(size**2)
    i = 0
    while i < size:
        j = 0
        while j < size:
            prod[size*i + j] += u[i] * v[j]
            j += 1
        i += 1
    return(prod)

def normalize(w):
    n = len(w)
    total = 0
    for i in range(n):
        total += w[i]
    return total / float(n)

def _stddev(v, m, unbias):
    n = len(v)
    s = 0.0
    for i in range(n):
        z = v[i] - m
        s += z * z
    if unbias == 0:
        return math.sqrt(s / float(n))
    else:
        return math.sqrt(s / float(n - 1))

def coskew(data, standardize=False, flatten=False, unbias=0):
    """Block-unfolded third cumulant tensor.
    
    Args:
      data: two-dimensional data matrix (signals = columns, samples = rows)
      rows: number of rows (samples per signal) in the data matrix
      cols: number of columns (signals) in the data matrix
    
    """
    rows = len(data)
    cols = len(data[0])
    tensor = array(cols)
    if flatten:
        pass
    else:
        if standardize:
            k = 0
            while k < cols:
                k_mean = 0
                r = 0
                while r < rows:
                    k_mean += data[r][k]
                    r += 1
                k_mean /= rows
                col = array(rows)
                for c in range(cols):
                    col[c] = data[c][k]
                k_std = _stddev(col, k_mean, unbias)
                face = array(cols)
                i = 0
                while i < cols:
                    i_mean = 0
                    r = 0
                    while r < rows:
                        i_mean += data[r][i]
                        r += 1
                    i_mean /= rows
                    col = array(rows)
                    for c in range(cols):
                        col[c] = data[c][i]
                    i_std = _stddev(col, i_mean, unbias)
                    face[i] = array(cols)
                    while j < cols:
                        j = 0
                        j_mean = 0
                        r = 0
                        while r < rows:
                            j_mean += data[r][j]
                            r += 1
                        j_mean /= rows
                        col = array(rows)
                        for c in range(cols):
                            col[c] = data[c][j]
                        j_std = _stddev(col, j_mean, unbias)
                        u = 0
                        row = 0
                        while row < rows:
                            i_center = data[row][i] - i_mean
                            j_center = data[row][j] - j_mean
                            k_center = data[row][k] - k_mean
                            u += i_center * j_center * k_center
                            row += 1
                        face[i][j] = u / (rows - unbias)
                        j += 1
                    tensor[k] = face
                    i += 1
                k += 1
            else:
                pass
    return(tensor)

def cokurt(data, standardize=False, flatten=False, unbias=0):
    """Block-unfolded fourth cumulant tensor.
    
    Args:
      data: two-dimensional data matrix (signals = columns, samples = rows)
      rows: number of rows (samples per signal) in the data matrix
      cols: number of columns (signals) in the data matrix
    
    """
    rows = len(data)
    cols = len(data[0])
    tensor = array(cols)
    if flatten:
        pass
    else:
        l = 0
        while l < cols:
            block = array(cols)
            k = 0
            while k < cols:
                face = array(cols)
                i = 0
                while i < cols:
                    j = 0
                    face[i] = array(cols)
                    while j < cols:
                        u = 0
                        row = 0
                        while row < rows:
                            i_mean = 0
                            j_mean = 0
                            k_mean = 0
                            l_mean = 0
                            r = 0
                            while r < rows:
                                i_mean += data[r][i]
                                j_mean += data[r][j]
                                k_mean += data[r][k]
                                l_mean += data[r][l]
                                r += 1
                            i_mean /= rows
                            j_mean /= rows
                            k_mean /= rows
                            l_mean /= rows
                            i_center = data[row][i] - i_mean
                            j_center = data[row][j] - j_mean
                            k_center = data[row][k] - k_mean
                            l_center = data[row][l] - l_mean
                            u += i_center * j_center * k_center * l_center
                            row += 1
                        face[i][j] = u / (rows - unbias)
                        j += 1
                    block[k] = face
                    i += 1
                tensor[l] = block
                k += 1
            l += 1
    return(tensor)

def coskewness(data, w=None, standardize=False, unbias=0):
    if w is None:
        cols = len(data[0])
        coskewness(data,
                   w=[1 for _ in range(cols)],
                   standardize=standardize,
                   unbias=unbias)
    if sum(w) != 1:
        w = normalize(w)
    S = matrix_multiply(matrix_multiply(transpose(w), coskew(data, standardize=standardize, flatten=True, unbias=unbias)), kron(w, w))
    return(S)

def cokurtosis(data, w=None, standardize=False, unbias=0):
    if w is None:
        cols = len(data[0])
        cokurtosis(data,
                   w=[1 for _ in range(cols)],
                   standardize=standardize,
                   unbias=unbias)
    if sum(w) != 1:
        w = normalize(w)
    S = matrix_multiply(matrix_multiply(transpose(w), cokurt(data, standardize=standardize, flatten=True, unbias=unbias)), kron(kron(w, w), w))
    return(S)
