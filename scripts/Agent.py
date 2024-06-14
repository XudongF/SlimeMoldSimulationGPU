import math
import numpy as np
import random
from numba import int32, float32, boolean
from numba.experimental import jitclass


class Slime:
    def __init__(self, location, angle, speed, energy_bar, live, sensordistance, sensorSize, sensorAngle, maxTurnAngle, move):
        self.location = location
        self.angle = angle
        self.speed = speed
        self.move = move
        self.sensordistance = sensordistance
        self.sensorSize = sensorSize
        self.sensorAngle = sensorAngle
        self.maxTurnAngle = maxTurnAngle
        self.energy_bar = energy_bar
        self.live = live

