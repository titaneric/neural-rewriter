import copy

import numpy as np

from models.data_utils.Seq import SeqManager


class TspNode:
    """
    Class to represent the node in 2d space
    """

    def __init__(self, x, y, embedding=None):
        self.x = x
        self.y = y

        self.px = None
        self.py = None
        self.pdis = None

        self.nx = None
        self.ny = None
        self.ndis = None

        self.embedding = None if embedding is None else embedding.copy()


class TspManager(SeqManager):
    def __init__(self):
        super().__init__()
        self.route = []
        self.tour = []
        self.opt_tour = []
        self.tot_dis = []
        self.encoder_outputs = None

    def clone(self):
        res = TspManager()
        res.nodes = []
        for i, node in enumerate(self.nodes):
            res.nodes.append(copy.copy(node))
        res.num_nodes = self.num_nodes
        res.route = self.route[:]
        res.tour = self.tour[:]
        res.tot_dis = self.tot_dis[:]
        res.encoder_outputs = self.encoder_outputs.clone()
        return res

    def get_dis(self, node_1, node_2):
        return np.sqrt((node_1.x - node_2.x) ** 2 + (node_1.y - node_2.y) ** 2)

    def get_neighbor_idxes(self, route_idx):
        """
            Get the neighbor indexes from the `route_idx` containing 
            all nodes expect `route_node_idx`
        """
        neighbor_idxes = []
        route_node_idx = self.tour[route_idx]
        pre_node_idx = self.tour[route_idx - 1]
        for i in range(len(self.tour)):
            cur_node_idx = self.tour[i]
            if route_node_idx == cur_node_idx:
                continue
            cur_node = self.get_node(cur_node_idx)
            neighbor_idxes.append(i)
        return neighbor_idxes

    def calc_tour_len(self):
        return sum(self.get_dis(self.get_node(n1), self.get_node(n2))
                   for n1, n2 in zip(self.tour, np.roll(self.tour, -1)))

    def update_node_info(self):
        """ 
            Only be called when the tour is completed
        """
        for i, (node_idx, next_node_idx, pre_node_idx) in enumerate(zip(
            self.tour, np.roll(self.tour, -1), np.roll(self.tour, 1))):
            node = self.get_node(node_idx)

            next_node = self.get_node(next_node_idx)
            ndis = self.get_dis(node, next_node)
            node.nx = next_node.x
            node.ny = next_node.y
            node.ndis = ndis

            pre_node = self.get_node(pre_node_idx)
            pdis = self.get_dis(node, pre_node)
            node.px = pre_node.x
            node.py = pre_node.y
            node.pdis = pdis

            node.embedding = [node.x, node.y,
                         node.px, node.py, node.pdis,
                         node.nx, node.ny, node.ndis]
            self.route[i] = node.embedding[:]

    def add_route_node(self, node_idx, insert_p):
        """
            Add the node into solution
        """
        # route stores the index embedding
        # tour stores the node index (solution)
        # tot_dis stores the distance change

        self.tour.insert(insert_p, node_idx)
        self.route.insert(insert_p, None)
        self.tot_dis.append(self.calc_tour_len())

