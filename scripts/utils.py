import numpy as np
import math
import random
import matplotlib.pyplot as plt
from itertools import compress
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
import pandas as pd
from numba import jit
import networkx as nx
from minimumCircle import welzl, Point
import itertools
import os
import scienceplots

plt.style.use("science")


def neighbors(shape):
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])


@jit(nopython=True)  # my mark
def mark(img, nbs):  # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        if img[p] == 0:
            continue
        s = 0
        for dp in nbs:
            if img[p + dp] != 0:
                s += 1
        if s == 2:
            img[p] = 1
        else:
            img[p] = 2


@jit(nopython=True)  # trans index to r, c...
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    rst -= 1
    return rst


@jit(nopython=True)  # fill a node (may be two or more points)
def fill(img, p, num, nbs, acc, buf):
    img[p] = num
    buf[0] = p
    cur = 0
    s = 1
    iso = True

    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if img[cp] == 2:
                img[cp] = num
                buf[s] = cp
                s += 1
            if img[cp] == 1:
                iso = False
        cur += 1
        if cur == s:
            break
    return iso, idx2rc(buf[:s], acc)


# trace the edge and use a buffer, then buf.copy, if use [] numba not works
@jit(nopython=True)
def trace(img, p, nbs, acc, buf):
    c1 = 0
    c2 = 0
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1 == 0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2 != 0:
            break
    return (c1 - 10, c2 - 10, idx2rc(buf[: cur + 1], acc))


@jit(nopython=True)  # parse the image then get the nodes and edges
def parse_struc(img, nbs, acc, iso, ring):
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in range(len(img)):
        if img[p] == 2:
            isiso, nds = fill(img, p, num, nbs, acc, buf)
            if isiso and not iso:
                continue
            num += 1
            nodes.append(nds)
    edges = []
    for p in range(len(img)):
        if img[p] < 10:
            continue
        for dp in nbs:
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
    if not ring:
        return nodes, edges
    for p in range(len(img)):
        if img[p] != 1:
            continue
        img[p] = num
        num += 1
        nodes.append(idx2rc([p], acc))
        for dp in nbs:
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges


# use nodes and edges build a networkx graph


def build_graph(nodes, edges, multi=False, full=True):
    os = np.array([i.mean(axis=0) for i in nodes])
    if full:
        os = os.round().astype(np.uint16)
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=os[i])
    for s, e, pts in edges:
        if full:
            pts[[0, -1]] = os[[s, e]]
        l = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
        graph.add_edge(s, e, pts=pts, weight=l)
    return graph


def mark_node(ske):
    buf = np.pad(ske, (1, 1), mode="constant").astype(np.uint16)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,) + buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    return buf


def build_sknw(ske, multi=False, iso=True, ring=True, full=True):
    buf = np.pad(ske, (1, 1), mode="constant").astype(np.uint16)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,) + buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    nodes, edges = parse_struc(buf, nbs, acc, iso, ring)
    return build_graph(nodes, edges, multi, full)


# draw the graph


def draw_graph(img, graph, cn=255, ce=128):
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for s, e in graph.edges():
        eds = graph[s][e]
        if isinstance(graph, nx.MultiGraph):
            for i in eds:
                pts = eds[i]["pts"]
                img[np.dot(pts, acc)] = ce
        else:
            img[np.dot(eds["pts"], acc)] = ce
    for idx in graph.nodes():
        pts = graph.nodes[idx]["pts"]
        img[np.dot(pts, acc)] = cn


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def generate_sample(Diameter, radius):
    r = radius * np.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    x = Diameter / 2 + r * math.cos(theta)
    y = Diameter / 2 + r * math.sin(theta)
    return np.rint(np.array([x, y], dtype=np.float32))


def plot_graph(G, node_dict):
    fig, ax = plt.subplots()
    pos = {i: G.nodes[i]["o"] for i in G.nodes}
    flipped_pos = {node: (y, x) for (node, (x, y)) in pos.items()}
    plot_nodes = [k for k in list(node_dict.keys()) if k in G.nodes]
    nx.draw_networkx(
        G,
        flipped_pos,
        node_color="red",
        nodelist=plot_nodes,
        edge_color="green",
        arrows=False,
        with_labels=False,
        node_size=5,
        ax=ax,
    )
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    ax = plt.gca()
    ax.invert_yaxis()
    plt.title("Simplified Graph")
    plt.show()


def plot_graph2(G):
    fig, ax = plt.subplots()

    for s, e in G.edges():
        ps = G[s][e]["pts"]
        ax.plot(ps[:, 1], ps[:, 0], "green")
    # draw node by o
    nodes = G.nodes()
    ps = np.array([nodes[i]["o"] for i in nodes])
    ax.plot(ps[:, 1], ps[:, 0], "r.", markersize=3)

    ax = plt.gca()
    ax.invert_yaxis()

    # title and show
    plt.title("Build Graph")
    plt.show()


def try_cluster(nodes, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(nodes)
    return kmeans.cluster_centers_, kmeans.labels_


def validate_number(nodes, buffer, threshold):
    for k in range(1, len(nodes) + 1):
        if k > threshold:
            print("The network is too sparse to simplify")
            centers = nodes
            labels = np.arange(len(nodes))
            break
        else:
            centers, labels = try_cluster(nodes, k)
            max_distance = np.max(
                [
                    math.dist(centers[idx], node)
                    for idx in range(k)
                    for node in list(compress(nodes, labels == idx))
                ]
            )
            if max_distance <= buffer:
                break
            else:
                continue
    return centers, labels


def consolidate_graph(G, buffer=10, threshold=200):
    # Remove dead nodes
    # dead_end_nodes = [node for node in G.nodes if G.degree[node] <= 1]
    # G.remove_nodes_from(dead_end_nodes)
    G.remove_edges_from(nx.selfloop_edges(G))

    # we use the k-means clustering to cluster the nodes to a buffer
    nodes_location = [G.nodes[node]["o"] for node in G.nodes]
    centers, labels = validate_number(nodes_location, buffer, threshold)
    # for fast getting node id
    nodes_dict = {G.nodes[i]["o"].tobytes(): i for i in G.nodes}
    numberOfNodes = max(G.nodes)
    for idx in range(len(centers)):
        new_id = idx + numberOfNodes + 1
        G.add_node(new_id, o=centers[idx])
        for node in list(compress(nodes_location, labels == idx)):
            node_id = nodes_dict[node.tobytes()]
            G = nx.contracted_nodes(G, new_id, node_id, copy=False, self_loops=False)
    return G


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def align_to_original(G, original_nodes):
    non_original = list(G.nodes)
    for node, pos in original_nodes.items():
        true_pos = pos[::-1]
        G.add_node(node, o=true_pos)
        node_dist = [math.dist(G.nodes[k]["o"], true_pos) for k in non_original]
        near_node = non_original[node_dist.index(min(node_dist))]
        if min(node_dist) < 100:
            non_original.remove(near_node)

            G = nx.contracted_nodes(G, node, near_node, copy=False, self_loops=False)

    return G


def align_to_origina_test(G, original_nodes):
    non_original = list(G.nodes)
    for node, pos in original_nodes.items():
        true_pos = pos[::-1]
        G.add_node(node, o=true_pos)

        tt = 50
        near_node = None
        for k in non_original:
            node_dist = math.dist(G.nodes[k]["o"], true_pos)
            if node_dist < tt:
                tt = node_dist
                near_node = k

        if near_node != None:
            non_original.remove(near_node)

            G = nx.contracted_nodes(G, node, near_node, copy=False, self_loops=False)

        else:
            tt = 50
            edge_1, edge_2 = None, None
            for u, v in G.edges():
                if (
                    angle_between(
                        G.nodes[u]["o"] - G.nodes[v]["o"], true_pos - G.nodes[v]["o"]
                    )
                    < math.pi / 2
                ) and (
                    angle_between(
                        G.nodes[v]["o"] - G.nodes[u]["o"], true_pos - G.nodes[u]["o"]
                    )
                    < math.pi / 2
                ):
                    dist_edge = np.linalg.norm(
                        np.abs(
                            np.cross(
                                G.nodes[v]["o"] - G.nodes[u]["o"],
                                G.nodes[u]["o"] - true_pos,
                            )
                        )
                    ) / np.linalg.norm(G.nodes[v]["o"] - G.nodes[u]["o"])
                    if dist_edge < tt:
                        tt = dist_edge
                        edge_1, edge_2 = u, v
                else:
                    continue

            if edge_1 is not None:
                G.remove_edge(edge_1, edge_2)
                G.add_edge(edge_1, node)
                G.add_edge(node, edge_2)

                plot_graph(
                    G,
                    node_dict={
                        node: original_nodes[node],
                        edge_1: G.nodes[edge_1]["o"],
                        edge_2: G.nodes[edge_1]["o"],
                    },
                )
                plt.savefig("results/test_edge.jpg", dpi=300)
                plt.show()

    return G


def draw_process(
    time_step, node_dict, binary=None, skele=None, simplified=None, save_path=None
):
    if binary is not None:
        fig, ax = plt.subplots()
        ax.imshow(binary, origin="upper")
        plt.title("Binary Image")
        plt.savefig(os.path.join(save_path, f"Binary{time_step}.jpg"), dpi=300)
        plt.show()
    if skele is not None:
        fig, ax = plt.subplots()
        plot_graph2(skele)
        plt.savefig(os.path.join(save_path, f"Skeleted{time_step}.jpg"), dpi=300)
        plt.show()
    if simplified is not None:
        fig, ax = plt.subplots()
        plot_graph(simplified, node_dict)
        plt.savefig(os.path.join(save_path, f"Simplified{time_step}.jpg"), dpi=300)
        plt.show()


def generate_network(
    img_data, buffer, diameter, boundaryControl, threshold=200, node_dict=None
):
    mask = create_circular_mask(
        diameter, diameter, radius=int(diameter / 2 - boundaryControl - 10)
    )

    img_data[~mask] = 0
    img_data = np.where(img_data > 55, 1, 0)

    ske = skeletonize(img_data).astype(np.uint16)
    G_ske = build_sknw(ske, iso=False, ring=False)

    G_consolidated = consolidate_graph(G_ske, buffer=buffer, threshold=threshold)

    G_consolidated.remove_nodes_from(list(nx.isolates(G_consolidated)))

    G_consolidated = align_to_original(G_consolidated, node_dict)
    return img_data, G_ske, G_consolidated


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum("...k,kl,...l->...", pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def getGaussianMap(mapSize, diffusionVariance, foodLocations, meanValue, mask):

    X = np.linspace(0, mapSize, mapSize)
    Y = np.linspace(0, mapSize, mapSize)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = np.zeros(X.shape, dtype=np.float32)

    for foodID in np.arange(len(foodLocations)):
        # we assume the food diffusion is independent in x and y direction, and the variance of x and y direction are same
        if len(diffusionVariance) > 1:
            diffusion = np.array(
                [[diffusionVariance[foodID], 0], [0, diffusionVariance[foodID]]]
            )

        else:
            diffusion = np.array([[diffusionVariance[0], 0], [0, diffusionVariance[0]]])

        Z += multivariate_gaussian(pos, np.array(foodLocations[foodID]), diffusion)

    # normalize the food peak source to footweight
    Z = meanValue * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    Z[~mask] = np.nan

    # Create a surface plot and projected filled contour plot under it.
    return Z


def get_network(file, boundaryControl):
    network_file = file.replace("node", "net")
    netfile = pd.read_csv(network_file, skiprows=8, sep="\t")

    netfile.init_node = "original" + netfile.init_node.astype(str)
    netfile.term_node = "original" + netfile.term_node.astype(str)

    pos = pd.read_csv(file, sep="\t")
    pos.Node = "original" + pos.Node.astype(str)
    pos[["X", "Y"]] = (pos[["X", "Y"]] - pos[["X", "Y"]].min(axis=0)) * 10000
    # # %%
    node_location = [
        np.rint(np.array((row.X, row.Y), dtype=np.float32)) for _, row in pos.iterrows()
    ]

    list_points = [Point(i[0], i[1]) for i in node_location]

    mec = welzl(list_points)

    diameter = (int(mec.R) + 4 * boundaryControl) * 2
    offset = int(diameter / 2 - max(mec.C.X, mec.C.Y))

    modified = [i + offset for i in node_location]

    node_dict = {pos.iloc[i].Node: modified[i] for i in range(len(modified))}
    return (
        diameter,
        node_dict,
        netfile,
    )


def measure_graph(G, node_dict):
    combinations = list(itertools.combinations(list(node_dict.keys()), 2))
    average_conn = []
    for comb in combinations:
        connectivity = nx.node_connectivity(G, comb[0], comb[1])
        average_conn.append(connectivity)

    total_length = 0
    for u, v in G.edges():
        edge_dist = math.dist(G.nodes[u]["o"], G.nodes[v]["o"])
        total_length += edge_dist

    average_network = nx.average_node_connectivity(G)
    return np.mean(average_conn), total_length, average_network
