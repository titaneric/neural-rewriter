# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import operator
import random
import time
import copy

from ..data_utils import data_utils


class tspRewriter(object):
	"""
	Rewriter for TSP.
	"""
	def move(self, dm, cur_route_idx, neighbor_route_idx):
		res = dm.clone()
		old_tour = res.tour[:]
		old_tour[cur_route_idx], old_tour[neighbor_route_idx] = old_tour[neighbor_route_idx], old_tour[cur_route_idx]
		res.tour = old_tour

		old_route = res.route[:]
		old_route[cur_route_idx], old_route[neighbor_route_idx] = old_route[neighbor_route_idx], old_route[cur_route_idx]
		res.route = old_route

		res.tot_dis = res.tot_dis[:]
		res.tot_dis.append(res.calc_tour_len())
		return res
