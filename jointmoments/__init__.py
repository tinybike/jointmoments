#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tensors and statistics for joint central moments.

"""
from __future__ import division

def array(size):
    return([0] * size)

def coskew(data, rows, cols, unbias):
    """Block-unfolded third cumulant tensor.
    
    Args:
      data: two-dimensional data matrix (signals = columns, samples = rows)
      rows: number of rows (samples per signal) in the data matrix
      cols: number of columns (signals) in the data matrix
    
    """
    tensor = array(cols)
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
                    r = 0
                    while r < rows:
                        i_mean += data[r][i]
                        j_mean += data[r][j]
                        k_mean += data[r][k]
                        r += 1
                    i_mean /= rows
                    j_mean /= rows
                    k_mean /= rows
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
    return(tensor)

def cokurt(data, rows, cols, unbias):
    """Block-unfolded fourth cumulant tensor.
    
    Args:
      data: two-dimensional data matrix (signals = columns, samples = rows)
      rows: number of rows (samples per signal) in the data matrix
      cols: number of columns (signals) in the data matrix
    
    """
    tensor = array(cols)
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

def coskewness(data, w):
    S = matrix_multiply(matrix_multiply(transpose(w), coskew(data)), kron(w, w))
    return(S)

def cokurtosis(data, w):
    S = matrix_multiply(matrix_multiply(transpose(w), cokurt(data)), kron(kron(w, w), w))
    return(S)
