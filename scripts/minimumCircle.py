# %%
# Python3 program to find the minimum enclosing
# circle for N integer points in a 2-D plane
from math import sqrt
from random import randint, shuffle

# Defining infinity
INF = 1e18

# Structure to represent a 2D point


class Point:
    def __init__(self, X=0, Y=0) -> None:
        self.X = X
        self.Y = Y

# Structure to represent a 2D circle


class Circle:
    def __init__(self, c=Point(), r=0) -> None:
        self.C = c
        self.R = r


# Function to return the euclidean distance
# between two points
def dist(a, b):
    return sqrt(pow(a.X - b.X, 2)
                + pow(a.Y - b.Y, 2))


# Function to check whether a point lies inside
# or on the boundaries of the circle
def is_inside(c, p):
    return dist(c.C, p) <= c.R


# The following two functions are used
# To find the equation of the circle when
# three points are given.

# Helper method to get a circle defined by 3 points
def get_circle_center(bx, by,
                      cx, cy):
    B = bx * bx + by * by
    C = cx * cx + cy * cy
    D = bx * cy - by * cx
    return Point((cy * B - by * C) / (2 * D),
                 (bx * C - cx * B) / (2 * D))

# Function to return the smallest circle
# that intersects 2 points


def circle_from1(A, B):
    # Set the center to be the midpoint of A and B
    C = Point((A.X + B.X) / 2.0, (A.Y + B.Y) / 2.0)

    # Set the radius to be half the distance AB
    return Circle(C, dist(A, B) / 2.0)

# Function to return a unique circle that
# intersects three points


def circle_from2(A, B, C):
    I = get_circle_center(B.X - A.X, B.Y - A.Y,
                          C.X - A.X, C.Y - A.Y)

    I.X += A.X
    I.Y += A.Y
    return Circle(I, dist(I, A))


# Function to check whether a circle
# encloses the given points
def is_valid_circle(c, P):

    # Iterating through all the points
    # to check  whether the points
    # lie inside the circle or not
    for p in P:
        if (not is_inside(c, p)):
            return False
    return True


# Function to return the minimum enclosing
# circle for N <= 3
def min_circle_trivial(P):
    assert(len(P) <= 3)
    if not P:
        return Circle()

    elif (len(P) == 1):
        return Circle(P[0], 0)

    elif (len(P) == 2):
        return circle_from1(P[0], P[1])

    # To check if MEC can be determined
    # by 2 points only
    for i in range(3):
        for j in range(i + 1, 3):

            c = circle_from1(P[i], P[j])
            if (is_valid_circle(c, P)):
                return c

    return circle_from2(P[0], P[1], P[2])


# Returns the MEC using Welzl's algorithm
# Takes a set of input points P and a set R
# points on the circle boundary.
# n represents the number of points in P
# that are not yet processed.
def welzl_helper(P, R, n):
    # Base case when all points processed or |R| = 3
    if (n == 0 or len(R) == 3):
        return min_circle_trivial(R)

    # Pick a random point randomly
    idx = randint(0, n-1)
    p = P[idx]

    # Put the picked point at the end of P
    # since it's more efficient than
    # deleting from the middle of the vector
    P[idx], P[n - 1] = P[n-1], P[idx]

    # Get the MEC circle d from the
    # set of points P - :p
    d = welzl_helper(P, R.copy(), n - 1)

    # If d contains p, return d
    if (is_inside(d, p)):
        return d

    # Otherwise, must be on the boundary of the MEC
    R.append(p)

    # Return the MEC for P - :p and R U :p
    return welzl_helper(P, R.copy(), n - 1)


def welzl(P):
    P_copy = P.copy()
    shuffle(P_copy)
    return welzl_helper(P_copy, [], len(P_copy))


if __name__ == '__main__':
    mec = welzl([Point(0, 0),
                 Point(0, 1),
                 Point(1, 0)])
    print("Center = {", mec.C.X, ",", mec.C.Y, "} Radius =", mec.R)

# https://www.geeksforgeeks.org/minimum-enclosing-circle-set-2-welzls-algorithm/
