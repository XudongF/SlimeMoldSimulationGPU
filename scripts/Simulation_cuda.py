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
            live=True,
            energy_bar=100,
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


def one_step_simpulation(
    slimes,
    petridish,
    occupied,
    foodlayer,
):
    dead_agent = []
    for agent in slimes:
        agent.update_location(petridish, occupied)
        agent.update_energy(foodlayer)
        agent.check_live()
        if not agent.live():
            dead_agent.append(agent)

    return [i for i in slimes if i not in dead_agent]


def evaporate(petridish):
    if petridish > 20:
        petridish = 20

    petridish -= 0.01
    return petridish


def set_zeros(occupied):
    occupied *= 0
    return occupied


def update_occupied(slimes, occupied):
    for agent in slimes:
        occupied[agent.location[0], agent.location[1]] = 255
    return occupied


def update_petridish(slimes, petridish):
    for slime in slimes:
        petridish[slime.location[0], slime.location[1]] += 0.5

    petridish = evaporate(petridish)

    return petridish


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

        slimes = one_step_simpulation(slimes, petridish, occupied, foodlayer)
        petridish = update_petridish(locations, petridish)
        evaporate[blockspergrid, threadsperblock](petridish_device)
        s += 1

    occupied_frame[0].save(
        f"results/agents.gif",
        format="GIF",
        append_images=occupied_frame[1:],
        save_all=True,
        duration=1,
        loop=0,
    )
