# %%
from Agent import Slime
import numpy as np
from utils import get_network, create_circular_mask, getGaussianMap
import random
from PIL import Image
import math
from tqdm import tqdm


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
        for _ in range(agent_number)
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


## some commet get somethign new


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
            for _ in range(foodNumber)
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
    for agent in tqdm(slimes):
        agent.update_location(petridish, occupied)
        agent.update_energy(foodlayer)
        agent.check_live()
        if not agent.live:
            dead_agent.append(agent)

    return [i for i in slimes if i not in dead_agent]


def evaporate(petridish):
    petridish[petridish > 20] = 20
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
    foodNumber = 9  # it won't be used in the future
    boundaryControl = 100
    diffusionK = np.ones((3, 3)) / 9
    hazardLocation = np.array([900, 1100], dtype=np.float32)
    hazardRange = 200
    withHazard = False
    location = "SiouxFalls"
    diameter, node_dict, _ = get_network(
        f"data/TNTPFiles/{location}/{location}_node.tntp", boundaryControl
    )
    print(f"the working environment diameter is {diameter}")

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

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(petridish.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(petridish.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    occupied_frame = []

    s = 0
    iterations = 10000
    while s < iterations:
        print("Nothing errors! start simulation!")
        set_zeros(occupied)
        occupied = update_occupied(slimes, occupied)
        if s % 50 == 0:
            print(f"******This is {s} of {iterations}*******")
            occupied_frame.append(draw_occupied(occupied, foodlayer))
        slimes = one_step_simpulation(slimes, petridish, occupied, foodlayer)
        petridish = update_petridish(slimes, petridish)
        evaporate(petridish)
        s += 1

    occupied_frame[0].save(
        "results/agents.gif",
        format="GIF",
        append_images=occupied_frame[1:],
        save_all=True,
        duration=1,
        loop=0,
    )
