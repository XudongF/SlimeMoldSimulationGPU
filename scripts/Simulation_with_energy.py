#!/bin/env python
# %%
from numba import cuda
import numpy as np
import random
from PIL import Image
import math
import imageio
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from utils import generate_sample, get_network, create_circular_mask, getGaussianMap

# Slime class definition
class Slime:
    def __init__(self, location, angle, speed, sensordistance, sensorSize, sensorAngle, maxTurnAngle, energy=1000):
        self.location = location
        self.angle = angle
        self.speed = speed
        self.sensordistance = sensordistance
        self.sensorSize = sensorSize
        self.sensorAngle = sensorAngle
        self.maxTurnAngle = maxTurnAngle
        self.energy = energy
        self.alive = True

def generate_sample(Diameter, radius):
    r = radius * np.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    x = Diameter / 2 + r * math.cos(theta)
    y = Diameter / 2 + r * math.sin(theta)
    return np.array([x, y], dtype=np.int32)

def generate_agents(diameter, boundaryControl, initial_speed, sensorDist, sensorSize, sensorAngle, maxTurnAngle, agent_number):
    locations = [generate_sample(diameter, radius=int(diameter / 2 - boundaryControl)) for _ in range(agent_number)]
    locations = np.unique(locations, axis=0)

    slimes = [Slime(location=location, speed=initial_speed, angle=random.uniform(0, 2 * math.pi),
                    sensordistance=sensorDist, sensorSize=sensorSize, sensorAngle=sensorAngle, maxTurnAngle=maxTurnAngle)
              for location in locations]

    return slimes

def generate_petridish(diameter):
    return np.zeros((diameter, diameter))

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
        foodLocation = [generate_sample(diameter, radius=int(diameter / 2 - 2 * boundaryControl)) for _ in range(foodNumber)]
    else:
        foodLocation = foodLocation

    foodlayer = getGaussianMap(mapSize=diameter, diffusionVariance=[200], foodLocations=foodLocation, meanValue=foodWeight, mask=mask)
    foodlayer[foodlayer < 0.1 * foodWeight] = 0

    return foodlayer

@cuda.jit
def agent_movement(agent_location, petridish, angle, maxTurnAngle, distance, occupied, sensorAngle, rng_states, energy, alive):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for agent_id in range(start, agent_location.shape[0], stride):
        if not alive[agent_id]:
            continue
        
        F = 0
        for i in range(12):
            for j in range(12):
                F += petridish[round(agent_location[agent_id][0] + distance *
                                     math.sin(angle[agent_id]) + i), round(agent_location[agent_id][1] + distance *
                                                                           math.cos(angle[agent_id]) + j)]
        FL = 0
        for i in range(12):
            for j in range(12):
                FL += petridish[round(agent_location[agent_id][0] + distance *
                                      math.sin(angle[agent_id]+sensorAngle) + i), round(agent_location[agent_id][1] + distance *
                                                                                        math.cos(angle[agent_id]+sensorAngle) + j)]
        FR = 0
        for i in range(12):
            for j in range(12):
                FR += petridish[round(agent_location[agent_id][0] + distance *
                                      math.sin(angle[agent_id]-sensorAngle) + i), round(agent_location[agent_id][1] + distance *
                                                                                        math.cos(angle[agent_id]-sensorAngle) + j)]

        if (F > FL) and (F > FR):
            angle[agent_id] = angle[agent_id] + 0
        elif (FL < FR):
            angle[agent_id] = angle[agent_id] - maxTurnAngle
        elif (FL > FR):
            angle[agent_id] = angle[agent_id] + maxTurnAngle
        else:
            angle[agent_id] = angle[agent_id] + 0

        next_location_x = round(
            agent_location[agent_id][0] + math.sin(angle[agent_id]))
        next_location_y = round(
            agent_location[agent_id][1] + math.cos(angle[agent_id]))

        if occupied[next_location_x, next_location_y] == 0:
            agent_location[agent_id][0] = next_location_x
            agent_location[agent_id][1] = next_location_y
        else:
            angle[agent_id] = xoroshiro128p_uniform_float32(rng_states, agent_id) * math.pi * 2

        energy[agent_id] -= 1
        if energy[agent_id] <= 0:
            alive[agent_id] = False

@cuda.jit
def evaporate(petridish):
    i, j = cuda.grid(2)
    if i < petridish.shape[0] and j < petridish.shape[1]:
        if petridish[i, j] > 20:
            petridish[i, j] = 20
        petridish[i, j] -= 0.01

@cuda.jit
def set_zeros_kernel(occupied):
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

def simulate(agents, petridish, foodlayer, max_steps, gif_filename="../results/agents_energy.gif"):
    frames = []
    
    agent_locations = np.array([agent.location for agent in agents], dtype=np.int32)
    angles = np.array([agent.angle for agent in agents], dtype=np.float32)
    rng_states = create_xoroshiro128p_states(len(agents), seed=1)
    occupied = np.zeros_like(petridish)
    energy = np.array([agent.energy for agent in agents], dtype=np.int32)
    alive = np.array([agent.alive for agent in agents], dtype=np.bool_)

    agent_locations_device = cuda.to_device(agent_locations)
    angles_device = cuda.to_device(angles)
    occupied_device = cuda.to_device(occupied)
    energy_device = cuda.to_device(energy)
    alive_device = cuda.to_device(alive)
    petridish_device = cuda.to_device(petridish)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(petridish.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(petridish.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    for step in range(max_steps):
        set_zeros_kernel[blockspergrid, threadsperblock](occupied_device)
        update_occupied[blockspergrid, threadsperblock](agent_locations_device, occupied_device)
        
        agent_movement[blockspergrid, threadsperblock](agent_locations_device, petridish_device, angles_device, np.pi/8, 1, occupied_device, np.pi/4, rng_states, energy_device, alive_device)
        
        update_petridish[blockspergrid, threadsperblock](agent_locations_device, petridish_device)
        evaporate[blockspergrid, threadsperblock](petridish_device)

        # Copy occupied back to host and capture the frame
        occupied_host = occupied_device.copy_to_host()
        img = draw_occupied(occupied_host, foodlayer)
        frames.append(img)
        
        if not any(alive_device.copy_to_host()):
            print(f"All agents have died by step {step}.")
            break

    # Save frames as a gif
    frames[0].save(gif_filename, format='GIF', append_images=frames[1:], save_all=True, duration=1, loop=0)
    return agents

# Parameters
foodNumber = 9
boundaryControl = 100
diffusionK = np.ones((3, 3)) / 9
hazardLocation = np.array([900, 1100], dtype=np.float32)
hazardRange = 200
withHazard = False
location = 'SiouxFalls'
diameter, node_dict, _ = get_network(f'../data/TNTPFiles/{location}/{location}_node.tntp', boundaryControl)

# Agent settings
agent_number = int(0.01 * 0.25 * 3.15 * diameter ** 2)
initial_speed = 1
sensorDist = 64
diffuseWeight = 5
sensorSize = 16
sensorAngle = math.pi / 4
maxTurnAngle = math.pi / 3
slimes = generate_agents(diameter, boundaryControl, initial_speed, sensorDist, sensorSize, sensorAngle, maxTurnAngle, agent_number)

mask = create_circular_mask(diameter, diameter, radius=int(diameter / 2 - boundaryControl))
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

# Initialize agent energy
agent_energy = np.full(locations.shape[0], 1000, dtype=np.float32)
agent_energy_device = cuda.to_device(agent_energy)

agent_alive = np.full(locations.shape[0], True, dtype=bool)
agent_alive_device = cuda.to_device(agent_alive)

threadsperblock = (32, 32)
blockspergrid_x = math.ceil(petridish.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(petridish.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
occupied_frames = []

s = 0
rng_states = create_xoroshiro128p_states(1024 * 1024, seed=1)
iterations = 10000

# Run the simulation and generate GIF
final_agents = simulate(slimes, petridish, foodlayer, iterations, gif_filename="../results/agents_energy.gif")

# Display the final state
img = draw_occupied(petridish, foodlayer)
img.show()
