import torch
import numpy as np
from scipy.spatial import distance_matrix

from models.data_utils.Tour import TspManager, TspNode

def load_ptr_dataset(file_type, num_nodes):
    filename = f"data/tsp_{num_nodes}_{file_type}_exact.txt"
    total_data = []

    with open(filename, "r") as f:
        for line in f.readlines():
            total_data.append(line)
    
    return total_data


def parse_line(line):
    nodes, tour = line.split("output")
    nodes = nodes.strip().split()
    tour = list(map(int, tour.strip().split()))

    graph = []
    dm = TspManager()
    for x, y in zip(nodes[::2], nodes[1::2]):
        x, y = float(x), float(y)
        graph.append([x, y])

        node = TspNode(x=x, y=y)
        dm.nodes.append(node)
    
    dm.num_nodes = len(dm.nodes)
    tour.pop()
    run_insertion(graph, "random", dm)
    dm.opt_tour = np.array(tour) - 1
    dm.update_node_info()

    return dm


"""
    Below insertion algorithm are directly copied from attention model
"""
def _calc_insert_cost(D, prv, nxt, ins):
    """
    Calculates insertion costs of inserting ins between prv and nxt
    :param D: distance matrix
    :param prv: node before inserted node, can be vector
    :param nxt: node after inserted node, can be vector
    :param ins: node to insert
    :return:
    """
    return (
        D[prv, ins]
        + D[ins, nxt]
        - D[prv, nxt]
    )


def run_insertion(loc, method, dm):
    n = len(loc)
    D = distance_matrix(loc, loc)

    mask = np.zeros(n, dtype=bool)
    # tour = []  # np.empty((0, ), dtype=int)
    for i in range(n):
        feas = mask == 0
        feas_ind = np.flatnonzero(mask == 0)
        if method == 'random':
            # Order of instance is random so do in order for deterministic results
            a = i
        elif method == 'nearest':
            if i == 0:
                a = 0  # order does not matter so first is random
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmin()] # node nearest to any in tour
        elif method == 'cheapest':
            assert False, "Not yet implemented" # try all and find cheapest insertion cost

        elif method == 'farthest':
            if i == 0:
                a = D.max(1).argmax()  # Node with farthest distance to any other node
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest
        mask[a] = True

        if not dm.tour:
            dm.add_route_node(a, 0)
        else:
            # Find index with least insert cost
            ind_insert = np.argmin(
                _calc_insert_cost(
                    D,
                    dm.tour,
                    np.roll(dm.tour, -1),
                    a
                )
            )
            dm.add_route_node(a, ind_insert + 1)

    # cost = D[dm.tour, np.roll(dm.tour, -1)].sum()
    # print(cost)
    # return cost, np.array(dm.tour)