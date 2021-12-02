import collections
from math import sqrt
import sys
import numpy as np


def calculate_diagonal(m):
    result = 1
    for i in range(min(len(m), len(m[0]))):
        if m[i][i] != 0:
            result = result * m[i][i]
    return result


def multiset_equal(x, y):
    first_set = collections.Counter(x)
    second_set = collections.Counter(y)
    return first_set == second_set


def find_max_element_after_zero(x):
    max_element = -sys.maxsize - 1
    x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
    for i in range(1, len(x)):
        if x[i - 1] == 0 and max_element < x[i]:
            max_element = x[i]
    return max_element


def convert_image(img, multiplier):
    num_channels = len(multiplier)
    h = len(img)
    w = len(img[0])
    res_img = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            for ch in range(num_channels):
                res_img[y][x] += multiplier[ch] * img[y][x][ch]
    return res_img


def run_length_encoding(x):
    elements = []
    counters = []
    if len(x) == 0:
        return elements, counters
    elements.append(x[0])
    counters.append(1)
    for elem in x[1:]:
        if elements[-1] == elem:
            counters[-1] += 1
        else:
            elements.append(elem)
            counters.append(1)
    return elements, counters


def pairwise_distance(x, y):
    matrix = [[0.0] * len(x) for _ in range(len(y))]
    for i in range(len(x)):
        for j in range(len(y)):
            vec1 = x[i]
            vec2 = y[j]
            for k in range(len(vec1)):
                matrix[i][j] += (vec1[k] - vec2[k]) ** 2
            matrix[i][j] = sqrt(matrix[i][j])
    return matrix