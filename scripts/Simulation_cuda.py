# %%
from Agent import Slime
from numba import cuda
import numpy as np
from utils import get_network, create_circular_mask, getGaussianMap
import random
from PIL import Image
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


def generate_sample(Diameter, radius):
    r = radius * np.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    x = Diameter / 2 + r * math.cos(theta)
    y = Diameter / 2 + r * math.sin(theta)
    return np.array([x, y], dtype=np.int32)


def generate_agents(
    diameter,
    boundaryControl,
    initial_speed,
    sensorDist,
    sensorSize,
    sensorAngle,
    maxTurnAngle,
    agent_number,
):
    locations = [
        generate_sample(diameter, radius=int(diameter / 2 - boundaryControl))
        for i in range(agent_number)
    ]
    locations = np.unique(locations, axis=0)

    slimes = [
        Slime(
            location=location,
            speed=initial_speed,
            angle=random.uniform(0, 2 * math.pi),
            sensordistance=sensorDist,
            sensorSize=sensorSize,
            sensorAngle=sensorAngle,
            maxTurnAngle=maxTurnAngle,
            move=True,
        )
        for location in locations
    ]

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
    img = Image.fromarray(plot_matrix.astype(np.uint8), "L")
    return img


def generate_Food(
    foodNumber,
    diameter,
    boundaryControl,
    foodWeight,
    mask,
    random_food=False,
    foodLocation=None,
):
    if random_food:
        random.seed(foodNumber)
        foodLocation = [
            generate_sample(diameter, radius=int(diameter / 2 - 2 * boundaryControl))
            for i in range(foodNumber)
        ]
    else:
        foodLocation = foodLocation

    foodlayer = getGaussianMap(
        mapSize=diameter,
        diffusionVariance=[200],
        foodLocations=foodLocation,
        meanValue=foodWeight,
        mask=mask,
    )

    foodlayer[foodlayer < 0.1 * foodWeight] = 0

    return foodlayer


@cuda.jit
def one_step_simpulation(
    agent_location,
    petridish,
    angle,
    maxTurnAngle,
    distance,
    occupied,
    sensorAngle,
    rng_states,
):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for agent_id in range(start, agent_location.shape[0], stride):
        F = 0
        for i in range(12):
            for j in range(12):
                F += petridish[
                    round(
                        agent_location[agent_id][0]
                        + distance * math.sin(angle[agent_id])
                        + i
                    ),
                    round(
                        agent_location[agent_id][1]
                        + distance * math.cos(angle[agent_id])
                        + j
                    ),
                ]
        FL = 0
        for i in range(12):
            for j in range(12):
                FL += petridish[
                    round(
                        agent_location[agent_id][0]
                        + distance * math.sin(angle[agent_id] + sensorAngle)
                        + i
                    ),
                    round(
                        agent_location[agent_id][1]
                        + distance * math.cos(angle[agent_id] + sensorAngle)
                        + j
                    ),
                ]
        FR = 0
        for i in range(12):
            for j in range(12):
                FR += petridish[
                    round(
                        agent_location[agent_id][0]
                        + distance * math.sin(angle[agent_id] - sensorAngle)
                        + i
                    ),
                    round(
                        agent_location[agent_id][1]
                        + distance * math.cos(angle[agent_id] - sensorAngle)
                        + j
                    ),
                ]

        if (F > FL) and (F > FR):
            angle[agent_id] = angle[agent_id] + 0
        elif FL < FR:
            angle[agent_id] = angle[agent_id] - maxTurnAngle
        elif FL > FR:
            angle[agent_id] = angle[agent_id] + maxTurnAngle
        else:
            angle[agent_id] = angle[agent_id] + 0

        next_location_x = round(agent_location[agent_id][0] + math.sin(angle[agent_id]))
        next_location_y = round(agent_location[agent_id][1] + math.cos(angle[agent_id]))

        if occupied[next_location_x, next_location_y] == 0:

            agent_location[agent_id][0] = next_location_x
            agent_location[agent_id][1] = next_location_y

        else:
            angle[agent_id] = (
                xoroshiro128p_uniform_float32(rng_states, agent_id) * math.pi * 2
            )


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


if __name__ == "__main__":
    # Env setting
    foodNumber = 9
    boundaryControl = 100
    diffusionK = np.ones((3, 3)) / 9
    hazardLocation = np.array([900, 1100], dtype=np.float32)
    hazardRange = 200
    withHazard = False
    location = "SiouxFalls"
    diameter, node_dict, _ = get_network(
        f"data/TNTPFiles/{location}/{location}_node.tntp", boundaryControl
    )

    # Agent setting
    agent_number = int(0.01 * 0.25 * 3.15 * diameter**2)
    initial_speed = 1
    sensorDist = 64
    diffuseWeight = 5
    sensorSize = 16
    sensorAngle = math.pi / 4
    maxTurnAngle = math.pi / 3
    slimes = generate_agents(
        diameter,
        boundaryControl,
        initial_speed,
        sensorDist,
        sensorSize,
        sensorAngle,
        maxTurnAngle,
        agent_number,
    )

    mask = create_circular_mask(
        diameter, diameter, radius=int(diameter / 2 - boundaryControl)
    )

    petridish = generate_petridish(diameter=diameter)

    locations = np.array([slime.location for slime in slimes])
    energy_bar = np.array([slime.energy_bar for slime in slimes])
    angles = np.array([slime.angle for slime in slimes])

    occupied = np.zeros((diameter, diameter), dtype=np.float32)
    occupied[~mask] = np.nan

    foodLocation = list(node_dict.values())
    foodlayer = generate_Food(
        foodNumber,
        diameter,
        boundaryControl,
        5,
        mask,
        random_food=False,
        foodLocation=foodLocation,
    )

    petridish = petridish + foodlayer

    petridish_device = cuda.to_device(petridish)
    occupied_device = cuda.to_device(occupied)
    angles_device = cuda.to_device(angles)
    locations_device = cuda.to_device(locations)
    energy_bar_device = cuda.to_device(energy_bar)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(petridish.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(petridish.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    occupied_frame = []

    s = 0
    rng_states = create_xoroshiro128p_states(1024 * 1024, seed=1)
    iterations = 10000
    while s < iterations:

        set_zeros[blockspergrid, threadsperblock](occupied_device)
        update_occupied[1024, 1024](locations_device, occupied_device)
        if s % 50 == 0:
            print(f"******This is {s} of {iterations}*******")
            occupied = occupied_device.copy_to_host()
            occupied_frame.append(draw_occupied(occupied, foodlayer))

        one_step_simpulation[1024, 1024](
            locations_device,
            petridish_device,
            angles_device,
            maxTurnAngle,
            sensorDist,
            occupied_device,
            sensorAngle,
            rng_states,
        )

        update_petridish[1024, 1024](locations_device, petridish_device)
        evaporate[blockspergrid, threadsperblock](petridish_device)
        s += 1

    occupied_frame[0].save(
        "results/agents.gif",
        format="GIF",
        append_images=occupied_frame[1:],
        save_all=True,
        duration=1,
        loop=0,
    )
