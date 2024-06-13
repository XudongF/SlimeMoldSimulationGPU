import math
import numpy as np
import random
from numba import int32, float32, boolean
from numba.experimental import jitclass


class Slime:
    def __init__(self, location, angle, speed, sensordistance, sensorSize, sensorAngle, maxTurnAngle, move):
        self.location = location
        self.angle = angle
        self.speed = speed
        self.move = move
        self.sensordistance = sensordistance
        self.sensorSize = sensorSize
        self.sensorAngle = sensorAngle
        self.maxTurnAngle = maxTurnAngle
        # self.energy = energy
        # self.alive = alive

# # Slime class definition
# class Slime:
#     def __init__(self, location, angle, speed, sensordistance, sensorSize, sensorAngle, maxTurnAngle, energy=1000):
#         self.location = location
#         self.angle = angle
#         self.speed = speed
#         self.sensordistance = sensordistance
#         self.sensorSize = sensorSize
#         self.sensorAngle = sensorAngle
#         self.maxTurnAngle = maxTurnAngle
#         self.energy = energy
#         self.alive = True


if __name__ == "__main__":
    slime = Slime(location=np.array([1, 1]), speed=1,
                  angle=random.uniform(0, math.pi), index=0, energy=1000)
    slime.update_location()
