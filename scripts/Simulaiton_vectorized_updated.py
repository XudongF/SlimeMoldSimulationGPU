#!/bin/env python
# %%
from Agent import Slime
from numba import cuda
import numpy as np
from utils import generate_sample, get_network, create_circular_mask, getGaussianMap
import random
from PIL import Image
import math
import time

def generate_sample(Diameter, radius):
    r = radius * np.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    x = Diameter/2 + r * math.cos(theta)
    y = Diameter/2 + r * math.sin(theta)
    return np.array([x, y], dtype=np.int32)

def generate_agents(diameter, boundaryControl, initial_speed, sensorDist, sensorSize, sensorAngle, maxTurnAngle, agent_number):
    locations = [generate_sample(diameter, radius=int(diameter/2 - boundaryControl)) for i in range(agent_number)]
    locations = np.unique(locations, axis=0)
    slimes = [Slime(location=location, speed=initial_speed, angle=random.uniform(0, 2*math.pi),
                    sensordistance=sensorDist, sensorSize=sensorSize, sensorAngle=sensorAngle, maxTurnAngle=maxTurnAngle, move=True) for location in locations]
    return slimes

def generate_petridish(diameter):
    petridish = np.zeros((diameter, diameter))
    return petridish

def draw_occupied(occupied, foodlayer=None):
    if foodlayer is not None:
        plot_matrix = 500 * foodlayer + occupied + 50
    else:
        plot_matrix = occupied + 50
    plot_matrix = np.where(plot_matrix > 255, 255, plot_matrix)
    img = Image.fromarray(plot_matrix.astype(np.uint8), 'L')
    return img

def generate_Food(foodNumber, diameter, boundaryControl, foodWeight, mask, random_food=False, foodLocation=None):
    if random_food:
        random.seed(foodNumber)
        foodLocation = [generate_sample(diameter, radius=int(diameter/2-2*boundaryControl)) for i in range(foodNumber)]
    else:
        foodLocation = foodLocation
    foodlayer = getGaussianMap(mapSize=diameter, diffusionVariance=[200], foodLocations=foodLocation, meanValue=foodWeight, mask=mask)
    foodlayer[foodlayer < 0.1*foodWeight] = 0
    return foodlayer

@cuda.jit
def evaporate(petridish):
    i, j = cuda.grid(2)
    if i < petridish.shape[0] and j < petridish.shape[1]:
        if petridish[i, j] > 20:
            petridish[i, j] = 20
        petridish[i, j] -= 0.01

@cuda.jit
def set_zeros(occupied):
    i, j = cuda.grid(2)
    if i < occupied.shape[0] and j < occupied.shape[1]:
        occupied[i, j] *= 0

@cuda.jit
def update_occupied(agent_location, occupied):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for location in range(start, agent_location.shape[0], stride):
        occupied[agent_location[location][0], agent_location[location][1]] = 255

@cuda.jit
def update_petridish(agent_location, petridish):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for location in range(start, agent_location.shape[0], stride):
        petridish[agent_location[location][0], agent_location[location][1]] += 0.5

@cuda.jit
def calculate_sensors(agent_location, petridish, angles, distance, sensorAngle):
    i = cuda.grid(1)
    if i < agent_location.shape[0]:
        x = agent_location[i, 0]
        y = agent_location[i, 1]
        angle = angles[i]

        x_front = np.clip(x + distance * np.sin(angle), 0, petridish.shape[0] - 1).astype(np.int32)
        y_front = np.clip(y + distance * np.cos(angle), 0, petridish.shape[1] - 1).astype(np.int32)

        x_left = np.clip(x + distance * np.sin(angle + sensorAngle), 0, petridish.shape[0] - 1).astype(np.int32)
        y_left = np.clip(y + distance * np.cos(angle + sensorAngle), 0, petridish.shape[1] - 1).astype(np.int32)

        x_right = np.clip(x + distance * np.sin(angle - sensorAngle), 0, petridish.shape[0] - 1).astype(np.int32)
        y_right = np.clip(y + distance * np.cos(angle - sensorAngle), 0, petridish.shape[1] - 1).astype(np.int32)

        F = petridish[x_front, y_front]
        FL = petridish[x_left, y_left]
        FR = petridish[x_right, y_right]

        # Store the sensor values in a shared memory array or global memory
        # for further processing

@cuda.jit
def update_angles(F, FL, FR, angles, maxTurnAngle):
    i = cuda.grid(1)
    if i < angles.shape[0]:
        angles[i] = np.where((F[i] > FL[i]) & (F[i] > FR[i]), angles[i], angles[i])
        angles[i] = np.where(FL[i] > FR[i], angles[i] + maxTurnAngle, angles[i])
        angles[i] = np.where(FR[i] > FL[i], angles[i] - maxTurnAngle, angles[i])

        # Add random noise to angles
        angles[i] += cuda.xoroshiro128p_uniform_float32(rng_states, i) * 2 * maxTurnAngle - maxTurnAngle

@cuda.jit
def update_positions(agent_location, angles, occupied, petridish_shape):
    i = cuda.grid(1)
    if i < agent_location.shape[0]:
        x = agent_location[i, 0]
        y = agent_location[i, 1]
        angle = angles[i]

        next_x = np.clip(x + np.sin(angle), 0, petridish_shape[0] - 1).astype(np.int32)
        next_y = np.clip(y + np.cos(angle), 0, petridish_shape[1] - 1).astype(np.int32)

        if occupied[next_x, next_y] == 0:
            agent_location[i, 0] = next_x
            agent_location[i, 1] = next_y

@cuda.jit
def one_step_simulation(agent_location, petridish, angles, maxTurnAngle, distance, occupied, sensorAngle, rng_states):
    num_agents = agent_location.shape[0]
    threadsperblock = (256, 1)  # Adjust this value based on your GPU's capabilities
    blockspergrid = (num_agents + threadsperblock[0] - 1) // threadsperblock[0]

    F, FL, FR = calculate_sensors[blockspergrid, threadsperblock](agent_location, petridish, angles, distance, sensorAngle)
    update_angles[blockspergrid, threadsperblock](F, FL, FR, angles, maxTurnAngle)
    update_positions[blockspergrid, threadsperblock](agent_location, angles, occupied, petridish.shape)

if __name__ == "__main__":
    # Env setting
    foodNumber = 9
    boundaryControl = 100
    diffusionK = np.ones((3, 3))/9
    hazardLocation = np.array([900, 1100], dtype=np.float32)
    hazardRange = 200
    withHazard = False
    location = 'SiouxFalls'
    diameter, node_dict, _ = get_network(f'../data/TNTPFiles/{location}/{location}_node.tntp', boundaryControl)

    # Agent setting
    agent_number = int(0.01*0.25*3.15*diameter**2)
    initial_speed = 1
    sensorDist = 64
    diffuseWeight = 5
    sensorSize = 16
    sensorAngle = math.pi/4
    maxTurnAngle = math.pi/3
    slimes = generate_agents(diameter, boundaryControl, initial_speed, sensorDist, sensorSize, sensorAngle, maxTurnAngle, agent_number)

    mask = create_circular_mask(diameter, diameter, radius=int(diameter/2-boundaryControl))
    petridish = generate_petridish(diameter=diameter)
    locations = np.array([slime.location for slime in slimes])
    angles = np.array([slime.angle for slime in slimes])

    occupied = np.zeros((diameter, diameter), dtype=np.float32)
    occupied[~mask] = np.nan

    foodLocation = list(node_dict.values())
    foodlayer = generate_Food(foodNumber, diameter, boundaryControl, 5, mask, random_food=False, foodLocation=foodLocation)
    petridish = petridish + foodlayer

    petridish_device = cuda.to_device(petridish)
    occupied_device = cuda.to_device(occupied)
    angles_device = cuda.to_device(angles)
    locations_device = cuda.to_device(locations)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(petridish.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(petridish.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    occupied_frame = []

    s = 0
    iterations = 10000
   
    from timeit import default_timer as timer
    start = timer()
    while s < iterations:
        set_zeros[blockspergrid, threadsperblock](occupied_device)
        update_occupied[(agent_number + (threadsperblock[0] - 1)) // threadsperblock[0], threadsperblock](locations_device, occupied_device)

        if s % 50 == 0:
            print(f"******This is {s} of {iterations}*******")
            occupied = occupied_device.copy_to_host()
            occupied_frame.append(draw_occupied(occupied, foodlayer))

        locations_host = locations_device.copy_to_host()
        angles_host = angles_device.copy_to_host()

        locations_host, angles_host = one_step_simulation(locations_host, petridish_device.copy_to_host(), angles_host, maxTurnAngle, sensorDist, occupied_device.copy_to_host(), sensorAngle)

        locations_device = cuda.to_device(locations_host)
        angles_device = cuda.to_device(angles_host)

        update_petridish[(agent_number + (threadsperblock[0] - 1)) // threadsperblock[0], threadsperblock](locations_device, petridish_device)
        evaporate[blockspergrid, threadsperblock](petridish_device)
        s += 1
    end = timer()
    elapsed = end - start
    print(f"Time for main loop: {elapsed}")
    
    occupied_frame[0].save(f'../results/agents.gif', format='GIF', append_images=occupied_frame[1:], save_all=True, duration=1, loop=0)
