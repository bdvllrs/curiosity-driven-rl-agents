"""
Taken from https://en.wikipedia.org/wiki/Maze_generation_algorithm
"""

import numpy
from numpy.random import randint as rand
import matplotlib.pyplot as pyplot


def maze(width=81, height=51, complexity=.75, density=.75):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))  # number of components
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # size of components
    # Build actual maze
    Z = numpy.ones(shape)
    # Fill borders
    Z[0, :] = Z[-1, :] = 0
    Z[:, 0] = Z[:, -1] = 0
    # Make aisles
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2  # pick a random position
        Z[y, x] = 0
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == 1:
                    Z[y_, x_] = 0
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 0
                    x, y = x_, y_
    return Z


if __name__ == "__main__":
    pyplot.figure(figsize=(10, 5))
    pyplot.imshow(maze(80, 40, complexity=0.7, density=1), cmap=pyplot.cm.binary, interpolation='nearest')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()
