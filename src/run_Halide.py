# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import numpy as np
import pandas as pd
import torch
import arguments
import models.data_utils.data_utils as data_utils
import models.model_utils as model_utils
from models.HalideModel import HalideModel


def create_model(args, term_vocab=None, term_vocab_list=None, op_vocab=None, op_vocab_list=None):
	model = HalideModel(args, term_vocab, term_vocab_list, op_vocab, op_vocab_list)
	if model.cuda_flag:
		model = model.cuda()
	model.share_memory()
	model_supervisor = model_utils.HalideSupervisor(model, args, term_vocab, term_vocab_list, op_vocab, op_vocab_list)
	if args.load_model:
		model_supervisor.load_pretrained(args.load_model)
	elif args.resume:
		pretrained = 'ckpt-' + str(args.resume).zfill(8)
		print('Resume from {} iterations.'.format(args.resume))
		model_supervisor.load_pretrained(args.model_dir+'/'+pretrained)
	else:
		print('Created model with fresh parameters.')
		model_supervisor.model.init_weights(args.param_init)
	return model_supervisor


def train(args):
	print('Training:')

	train_data = data_utils.load_dataset(args.train_dataset, args)
	eval_data = data_utils.load_dataset(args.val_dataset, args)

	DataProcessor = data_utils.HalideDataProcessor()

	if args.train_proportion < 1.0:
		random.shuffle(train_data)
		train_data_size = int(train_data_size * args.train_proportion)
		train_data = train_data[:train_data_size]

	if args.train_max_len is not None:
		train_data = DataProcessor.prune_dataset(train_data, max_len=args.train_max_len)

	train_data_size = len(train_data)
	term_vocab, term_vocab_list = DataProcessor.load_term_vocab()
	op_vocab, op_vocab_list = DataProcessor.load_ops()
	args.term_vocab_size = len(term_vocab)
	args.op_vocab_size = len(op_vocab)
	model_supervisor = create_model(args, term_vocab, term_vocab_list, op_vocab, op_vocab_list)

	if args.resume:
		resume_step = True
	else:
		resume_step = False
	resume_idx = args.resume * args.batch_size

	logger = model_utils.Logger(args)
	if args.resume:
		logs = pd.read_csv("../logs/" + args.log_name)
		for index, log in logs.iterrows():
			val_summary = {'avg_reward': log['avg_reward'], 'global_step': log['global_step']}
			logger.write_summary(val_summary)

	for epoch in range(resume_idx//train_data_size, args.num_epochs):
		random.shuffle(train_data)
		for batch_idx in range(0+resume_step*resume_idx%train_data_size, train_data_size, args.batch_size):
			resume_step = False
			print(epoch, batch_idx)
			batch_data = DataProcessor.get_batch(train_data, args.batch_size, batch_idx)
			train_loss, train_reward = model_supervisor.train(batch_data)
			print('train loss: %.4f train reward: %.4f' % (train_loss, train_reward))

			if model_supervisor.global_step % args.eval_every_n == 0:
				eval_loss, eval_reward = model_supervisor.eval(eval_data, args.output_trace_flag, args.max_eval_size)
				val_summary = {'avg_reward': eval_reward, 'global_step': model_supervisor.global_step}
				logger.write_summary(val_summary)
				model_supervisor.save_model()

			if args.lr_decay_steps and model_supervisor.global_step % args.lr_decay_steps == 0:
				model_supervisor.model.lr_decay(args.lr_decay_rate)
				if model_supervisor.model.cont_prob > 0.01:
					model_supervisor.model.cont_prob *= 0.5


def evaluate(args):
	print('Evaluation:')

	test_data = data_utils.load_dataset(args.test_dataset, args)
	test_data_size = len(test_data)

	args.dropout_rate = 0.0

	DataProcessor = data_utils.HalideDataProcessor()

	if args.test_min_len is not None:
		test_data = DataProcessor.prune_dataset(test_data, min_len=args.test_min_len)

	term_vocab, term_vocab_list = DataProcessor.load_term_vocab()
	op_vocab, op_vocab_list = DataProcessor.load_ops()
	args.term_vocab_size = len(term_vocab)
	args.op_vocab_size = len(op_vocab)
	model_supervisor = create_model(args, term_vocab, term_vocab_list, op_vocab, op_vocab_list)
	test_loss, test_reward = model_supervisor.eval(test_data, args.output_trace_flag, args.output_trace_option, args.output_trace_file)

	print('test loss: %.4f test reward: %.4f' % (test_loss, test_reward))


if __name__ == "__main__":
	argParser = arguments.get_arg_parser("Halide")
	args = argParser.parse_args()
	args.cuda = not args.cpu and torch.cuda.is_available()
	random.seed(args.seed)
	np.random.seed(args.seed)
	if args.eval:
		evaluate(args)
	else:
		train(args)
