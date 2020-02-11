import numpy as np

from models.data_utils.Seq import SeqManager


class TspNode:
    """
    Class to represent the node in 2d space
    """

    def __init__(self, x, y, px, py, dis, embedding=None):
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.dis = dis
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
            res.nodes.append(TspNode(x=node.x, y=node.y, px=node.px,
                                     py=node.py, dis=node.dis, embedding=node.embedding))
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
        return sum(self.get_dis(self.get_node(n1), self.get_node(n2)) \
            for n1, n2 in zip(self.tour, np.roll(self.tour, -1)))

    def add_route_node(self, node_idx, insert_p):
        """
            Add the node into solution
            Node embedding is
            [node.x, node.y, pre_node.x, pre_node.y, dis(node, pre_node)]
        """
        # route stores the node embedding
        # tour stores the node index (solution)
        # tot_dis stores the distance change

        pre_insert = (insert_p - 1) % self.num_nodes
        # next_insert = (insert_p + 1) % self.num_nodes

        # def calc_diff(origin_dis, pre_insert, next_insert):
        #     new_dis = self.get_dis(pre_insert, insert_p) + \
        #         self.get_dis(insert_p, next_insert)
        #     return new_dis - origin_dis

        node = self.get_node(node_idx)
        if not self.tour:
            pre_node_idx = 0
        else:
            pre_node_idx = self.tour[pre_insert]
        pre_node = self.get_node(pre_node_idx)
        self.tour.insert(insert_p, node_idx)
        cur_dis = self.get_dis(node, pre_node)
        # if not self.tot_dis:
        #     self.tot_dis.append(cur_dis)
        # else:
        #     self.tot_dis.append(self.tot_dis[-1] + calc_diff(cur_dis, pre_insert, next_insert))
        self.tot_dis.append(self.calc_tour_len())
        new_node = TspNode(x=node.x, y=node.y,
                           px=pre_node.x, py=pre_node.y, dis=cur_dis)
        new_node.embedding = [new_node.x, new_node.y,
                              new_node.px, new_node.py, new_node.dis]
        self.nodes[node_idx] = new_node
        # self.route.append(new_node.embedding[:])
        self.route.insert(insert_p, new_node.embedding[:])
