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
from multiprocessing.pool import ThreadPool

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .data_utils import data_utils
from .modules import tspInputEncoder, mlp
from .rewriter import tspRewriter
from .BaseModel import BaseModel

eps = 1e-3
log_eps = np.log(eps)


class tspModel(BaseModel):
	"""
	Model architecture for vehicle routing.
	"""
	def __init__(self, args):
		super(tspModel, self).__init__(args)
		self.input_format = args.input_format
		self.embedding_size = args.embedding_size
		self.attention_size = args.attention_size
		self.sqrt_attention_size = int(np.sqrt(self.attention_size))
		self.reward_thres = -0.01
		self.input_encoder = tspInputEncoder.TourLSTM(args)
		self.policy_embedding = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 4 + self.embedding_size, self.MLP_hidden_size, self.attention_size, self.cuda_flag, self.dropout_rate)

		"""
			value estimator and policy input size is only self.LSTM_hidden_size * 2
			because it contains only cur_node embedding
			x2 due to bi-directional LSTM
		"""
		self.policy = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 2, self.MLP_hidden_size, self.attention_size, self.cuda_flag, self.dropout_rate)
		self.value_estimator = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 2, self.MLP_hidden_size, 1, self.cuda_flag, self.dropout_rate)
		self.rewriter = tspRewriter()

		if args.optimizer == 'adam':
			self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
		elif args.optimizer == 'sgd':
			self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
		elif args.optimizer == 'rmsprop':
			self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
		else:
			raise ValueError('optimizer undefined: ', args.optimizer)


	def rewrite(self, dm, trace_rec, candidate_rewrite_pos, eval_flag, max_search_pos, reward_thres=None):

		# Sort the `candidate_rewrite_pos` by estimated value (The more tour difference is, the higher value is.)
		candidate_rewrite_pos.sort(reverse=True, key=operator.itemgetter(0))
		if not eval_flag:
			sample_exp_reward_tensor = []
			for idx, (cur_pred_reward, cur_pred_reward_tensor, rewrite_pos) in enumerate(candidate_rewrite_pos):
				sample_exp_reward_tensor.append(cur_pred_reward_tensor)
			sample_exp_reward_tensor = torch.cat(sample_exp_reward_tensor, 0)
			sample_exp_reward_tensor = torch.exp(sample_exp_reward_tensor * 10)
			sample_exp_reward = sample_exp_reward_tensor.data.cpu()

		# `candidate_dm` stores the new dm
		candidate_dm = []
		# `candidate_rewrite_rec` stores the neccesary info to update the policy
		candidate_rewrite_rec = []
		candidate_trace_rec = []
		candidate_scores = []

		if not eval_flag:
			sample_rewrite_pos_dist = Categorical(sample_exp_reward_tensor)
			sample_rewrite_pos = sample_rewrite_pos_dist.sample(sample_shape=[len(candidate_rewrite_pos)])
			#sample_rewrite_pos = torch.multinomial(sample_exp_reward_tensor, len(candidate_rewrite_pos))
			sample_rewrite_pos = sample_rewrite_pos.data.cpu().numpy()
			indexes = np.unique(sample_rewrite_pos, return_index=True)[1]
			sample_rewrite_pos = [sample_rewrite_pos[i] for i in sorted(indexes)]
			sample_rewrite_pos = sample_rewrite_pos[:self.num_sample_rewrite_pos]
			sample_exp_reward = [sample_exp_reward[i] for i in sample_rewrite_pos]
			sample_rewrite_pos = [candidate_rewrite_pos[i] for i in sample_rewrite_pos]
		else:
			sample_rewrite_pos = candidate_rewrite_pos.copy()

		# Start to use the picked region to estimate the next swapped area
		for idx, (pred_reward, cur_pred_reward_tensor, rewrite_pos) in enumerate(sample_rewrite_pos):
			# only search max_search_pos=1 times
			if len(candidate_dm) > 0 and idx >= max_search_pos:
				break
			if reward_thres is not None and pred_reward < reward_thres:
				if eval_flag:
					break
				elif np.random.random() > self.cont_prob:
					continue
			# `candidate_neighbor_idxes` is the neighbor indexes on the tour
			candidate_neighbor_idxes = dm.get_neighbor_idxes(rewrite_pos)
			cur_node_idx = dm.tour[rewrite_pos]
			cur_node = dm.get_node(cur_node_idx)
			pre_node_idx = dm.tour[rewrite_pos - 1]
			pre_node = dm.get_node(pre_node_idx)
			# cur_state is the current rewrite_pos embedding
			cur_state = dm.encoder_outputs[rewrite_pos].unsqueeze(0)
			# cur_states_1 stores the cur_state embedding
			cur_states_1 = []
			# cur_states_2 stores the neighbor embedding
			cur_states_2 = []
			# new embeddings contains the neighbor coords, pre_node coords and distance between them
			new_embeddings = []
			for i in candidate_neighbor_idxes:
				neighbor_idx = dm.tour[i]
				neighbor_node = dm.get_node(neighbor_idx)
				cur_states_1.append(cur_state.clone())
				cur_states_2.append(dm.encoder_outputs[i].unsqueeze(0))
				new_embedding = [neighbor_node.x, neighbor_node.y, pre_node.x, pre_node.y, dm.get_dis(pre_node, neighbor_node)]
				new_embeddings.append(new_embedding[:])				
			cur_states_1 = torch.cat(cur_states_1, 0)
			cur_states_2 = torch.cat(cur_states_2, 0)
			new_embeddings = data_utils.np_to_tensor(new_embeddings, 'float', self.cuda_flag)
			# Use the current rewrite pos, neighbor and other embedding to estimate the policy
			policy_inputs = torch.cat([cur_states_1, cur_states_2, new_embeddings], 1)
			# `ctx_embeddings` is the key (K) that attention want to point to (next swapped position)
			ctx_embeddings = self.policy_embedding(policy_inputs)
			# `cur_state_key` is the query (Q) 
			cur_state_key = self.policy(torch.cat([cur_state], dim=1))
			# `ac_logits` is the compactibility
			ac_logits = torch.matmul(cur_state_key, torch.transpose(ctx_embeddings, 0, 1)) / self.sqrt_attention_size
			ac_logprobs = nn.LogSoftmax()(ac_logits)
			# `ac_probs` is the attention weight or pointer
			ac_probs = nn.Softmax()(ac_logits)
			ac_logits = ac_logits.squeeze(0)
			ac_logprobs = ac_logprobs.squeeze(0)
			ac_probs = ac_probs.squeeze(0)
			# Sample or argmax the candidate_acs (next swapped position)
			if eval_flag:
				_, candidate_acs = torch.sort(ac_logprobs, descending=True)
				candidate_acs = candidate_acs.data.cpu().numpy()
			else:
				candidate_acs_dist = Categorical(ac_probs)
				candidate_acs = candidate_acs_dist.sample(sample_shape=[ac_probs.size()[0]])
				#candidate_acs = torch.multinomial(ac_probs, ac_probs.size()[0])
				candidate_acs = candidate_acs.data.cpu().numpy()
				indexes = np.unique(candidate_acs, return_index=True)[1]
				candidate_acs = [candidate_acs[i] for i in sorted(indexes)]
	
			for i in candidate_acs:
				neighbor_idx = candidate_neighbor_idxes[i]
				# Start to rewrite the instance
				new_dm = self.rewriter.move(dm, rewrite_pos, neighbor_idx)
				# print(dm.tour, "->", new_dm.tour)
				# If lastest distance doesn't change, ignore it
				if new_dm.tot_dis[-1] in trace_rec:
					continue
				candidate_dm.append(new_dm)
				candidate_rewrite_rec.append((ac_logprobs, pred_reward, cur_pred_reward_tensor, rewrite_pos, i, new_dm.tot_dis[-1]))
				if len(candidate_dm) >= max_search_pos:
					break
		
		return candidate_dm, candidate_rewrite_rec


	def batch_rewrite(self, dm, trace_rec, candidate_rewrite_pos, eval_flag, max_search_pos, reward_thres):
		candidate_dm = []
		candidate_rewrite_rec = []
		for i in range(len(dm)):
			cur_candidate_dm, cur_candidate_rewrite_rec = self.rewrite(dm[i], trace_rec[i], candidate_rewrite_pos[i], eval_flag, max_search_pos, reward_thres)
			candidate_dm.append(cur_candidate_dm)
			candidate_rewrite_rec.append(cur_candidate_rewrite_rec)
		return candidate_dm, candidate_rewrite_rec


	def forward(self, batch_data, eval_flag=False):
		torch.set_grad_enabled(not eval_flag)
		dm_list = []
		batch_size = len(batch_data)
		for dm in batch_data:
			dm_list.append(dm)
		dm_list = self.input_encoder.calc_embedding(dm_list, eval_flag)

		active = True
		reduce_steps = 0

		# `trace_rec` contains the lastest distance for every instance
		trace_rec = [{} for _ in range(batch_size)]
		# `rewrite_rec` stores the new rewrite position for every instance
		rewrite_rec = [[] for _ in range(batch_size)]
		# `dm_rec` containes every instace
		dm_rec = [[] for _ in range(batch_size)]

		""" 
			Push the current instance into `dm_rec`
			which can be used in the reward calculation
		"""
		for idx in range(batch_size):
			dm_rec[idx].append(dm_list[idx])
			trace_rec[idx][dm_list[idx].tot_dis[-1]] = 0

		# self.max_reduce_steps = 3
		while active and (self.max_reduce_steps is None or reduce_steps < self.max_reduce_steps):
			active = False
			reduce_steps += 1
			# `node_idxes` stores the tuple of instance index and tour index (rewrite position)
			node_idxes = []
			# `node_states` stores the node embedding
			node_states = []
			"""
				Prepare batch-size node embeddings
				Resume it after feeding to value estimator
			"""
			for dm_idx in range(batch_size):
				dm = dm_list[dm_idx]
				for i in range(len(dm.tour)):
					cur_node_idx = dm.tour[i]
					cur_node = dm.get_node(cur_node_idx)
					node_idxes.append((dm_idx, i))
					node_states.append(dm.encoder_outputs[i].unsqueeze(0))
			pred_rewards = []
			for st in range(0, len(node_idxes), self.batch_size):
				cur_node_states = node_states[st: st + self.batch_size]
				cur_node_states = torch.cat(cur_node_states, 0)
				"""
					For current value estimator, we only use the node embedding because
					TSP have no depot information.
				"""
				cur_pred_rewards = self.value_estimator(torch.cat([cur_node_states], dim=1))
				pred_rewards.append(cur_pred_rewards)
			pred_rewards = torch.cat(pred_rewards, 0)

			# For every instance, append the node index and corresponding estimated value
			candidate_rewrite_pos = [[] for _ in range(batch_size)]

			for idx, (dm_idx, node_idx) in enumerate(node_idxes):
				candidate_rewrite_pos[dm_idx].append((pred_rewards[idx].data[0], pred_rewards[idx], node_idx))

			# Batch size rewrite
			candidate_dm, candidate_rewrite_rec = self.batch_rewrite(dm_list, trace_rec, candidate_rewrite_pos, eval_flag, max_search_pos=1, reward_thres=self.reward_thres)
			for dm_idx in range(batch_size):
				cur_candidate_dm = candidate_dm[dm_idx]
				cur_candidate_rewrite_rec = candidate_rewrite_rec[dm_idx]
				if len(cur_candidate_dm) > 0:
					active = True
					cur_dm = cur_candidate_dm[0]
					cur_rewrite_rec = cur_candidate_rewrite_rec[0]
					# Update the instance, rewrite position and lastest tour length
					dm_list[dm_idx] = cur_dm
					rewrite_rec[dm_idx].append(cur_rewrite_rec)
					trace_rec[dm_idx][cur_dm.tot_dis[-1]] = 0
			if not active:
				break
			
			# Use the updated instance to obatin new tour embedding
			updated_dm = self.input_encoder.calc_embedding(dm_list, eval_flag)
			for i in range(batch_size):
				# Filter the instance that having the same distance
				if updated_dm[i].tot_dis[-1] != dm_rec[i][-1].tot_dis[-1]:
					dm_rec[i].append(updated_dm[i])

		total_policy_loss = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)
		total_value_loss = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)

		pred_value_rec = []
		value_target_rec = []
		total_reward = 0
		total_rewrite_steps = 0
		for dm_idx, cur_dm_rec in enumerate(dm_rec):
			pred_dis = []
			for dm in cur_dm_rec:
				pred_dis.append(dm.tot_dis[-1])
			# cur_dm_rec contains the initial solution in the first place
			best_reward = pred_dis[0]
			
			for idx, (ac_logprob, pred_reward, cur_pred_reward_tensor, rewrite_pos, applied_op, new_dis) in enumerate(rewrite_rec[dm_idx]):
				cur_reward = pred_dis[idx] - pred_dis[idx + 1]
				best_reward = min(best_reward, pred_dis[idx + 1])

				if self.gamma > 0.0:
					decay_coef = 1.0
					num_rollout_steps = len(cur_dm_rec) - idx - 1
					for i in range(idx + 1, idx + 1 + num_rollout_steps):
						cur_reward = max(decay_coef * (pred_dis[idx] - pred_dis[i]), cur_reward)
						# print(pred_dis, idx, i, cur_reward)
						decay_coef *= self.gamma

				cur_reward_tensor = data_utils.np_to_tensor(np.array([cur_reward], dtype=np.float32), 'float', self.cuda_flag, volatile_flag=True)
				if cur_reward - pred_reward > 0:
					ac_mask = np.zeros(ac_logprob.size()[0])
					ac_mask[applied_op] = cur_reward - pred_reward
					ac_mask = data_utils.np_to_tensor(ac_mask, 'float', self.cuda_flag, eval_flag)
					total_policy_loss -= ac_logprob[applied_op] * ac_mask[applied_op]
				pred_value_rec.append(cur_pred_reward_tensor)
				value_target_rec.append(cur_reward_tensor)
			
			total_reward += best_reward

		if len(pred_value_rec) > 0:
			pred_value_rec = torch.cat(pred_value_rec, 0)
			value_target_rec = torch.cat(value_target_rec, 0)
			pred_value_rec = pred_value_rec.unsqueeze(1)
			value_target_rec = value_target_rec.unsqueeze(1)
			total_value_loss = F.smooth_l1_loss(pred_value_rec, value_target_rec, size_average=False)
		total_policy_loss /= batch_size
		total_value_loss /= batch_size
		total_loss = total_policy_loss * self.value_loss_coef + total_value_loss
		total_reward = total_reward * 1.0 / batch_size

		return total_loss, total_reward, dm_rec


