import math
import random


class Slime:
    def __init__(
        self,
        location,
        angle,
        speed,
        sensordistance,
        sensorSize,
        sensorAngle,
        maxTurnAngle,
        move,
        energy_bar,
        live,
    ):
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

    def update_energy(self, foodlayer):
        if foodlayer[self.location[0]][self.location[1]] == 5:
            self.energy_bar = 10
        else:
            self.energy_bar -= 1

    def check_live(self):
        if self.energy_bar > 0:
            self.live = True
        else:
            self.live = False

    def update_location(self, petridish, occupied):
        F = 0
        for i in range(12):
            for j in range(12):
                F += petridish[
                    round(
                        self.location[0]
                        + self.sensordistance * math.sin(self.angle)
                        + i
                    ),
                    round(
                        self.location[1]
                        + self.sensordistance * math.cos(self.angle)
                        + j
                    ),
                ]
        FL = 0
        for i in range(12):
            for j in range(12):
                FL += petridish[
                    round(
                        self.location[0]
                        + self.sensordistance * math.sin(self.angle + self.sensorAngle)
                        + i
                    ),
                    round(
                        self.location[1]
                        + self.sensordistance * math.cos(self.angle + self.sensorAngle)
                        + j
                    ),
                ]
        FR = 0
        for i in range(12):
            for j in range(12):
                FR += petridish[
                    round(
                        self.location[0]
                        + self.sensordistance * math.sin(self.angle - self.sensorAngle)
                        + i
                    ),
                    round(
                        self.location[1]
                        + self.sensordistance * math.cos(self.angle - self.sensorAngle)
                        + j
                    ),
                ] 

        if (F > FL) and (F > FR):
            self.angle += 0
        elif FL < FR:
            self.angle = self.angle - self.maxTurnAngle
        elif FL > FR:
            self.angle = self.angle + self.maxTurnAngle
        else:
            self.angle = +0

        temp_x = int(self.location[0] + math.sin(self.angle))
        temp_y = int(self.location[1] + math.cos(self.angle))
        if occupied[temp_x, temp_y] == 0 :
            self.location[0] = temp_x
            self.location[1] = temp_y
        else:
            self.angle = random.uniform(0, math.pi * 2)
