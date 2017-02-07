#!/bin/python2

import numpy as np
import matplotlib.pyplot as plt
import MLPApproach

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def separate_projectiles(data):
    data2 = np.delete(data, 0, axis=1) # delete col 1
    n = len(data2)
    res = []
    for i in range(n):
        curr = data2[i]*100
        if (curr == [0, 0]).all():
            res.append([curr])
        else:
            res[-1].append(curr)
    return np.array(res)

# visualize projectile curve
def show_projectile(points):
    np_points = np.array(points)
    plt.plot(np_points[:, 0], np_points[:, 1], linewidth=1.4)

# shuffle data
def shuffle(data):
    return np.random.permutation(data)

# fit a quadratic curve for given sample projectile
def get_curve_params(example):
    example = np.array(example)
    return np.polyfit(example[:, 0], example[:, 1], 2)

# plot quadratic curve
def show_curve(curve_params, x_max, color = 'g--'):
    x = np.linspace(0, x_max, 100)
    y = curve_params[0]*(x**2)+curve_params[1]*x+curve_params[2]
    y = [el for el in y if el > 0]
    x = x[:len(y)]
    plt.plot(x, y, color)

data = np.loadtxt(open("projectiles.csv", "rb"), delimiter=",").astype(np.float32)
examples = separate_projectiles(data)
MLPApproach.do(examples, get_curve_params, shuffle, show_projectile, show_curve)
