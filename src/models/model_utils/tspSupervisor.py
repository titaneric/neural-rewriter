import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from models.model_utils import Supervisor

class tspSupervisor(Supervisor):
    def __init__(self, model, args):
        super().__init__(model, args)

    def train(self, batch_data):
        self.model.optimizer.zero_grad()
        avg_loss, avg_reward, dm_rec = self.model(batch_data)
        self.global_step += 1
        avg_loss.backward()
        self.model.train()
        return avg_loss.item(), avg_reward


