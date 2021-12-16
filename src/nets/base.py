import torch
import torch.nn as nn

class TrainWrapperBaseClass():
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(self.args.gpu)
        self.global_step = 0

    def init_optimizer(self) -> None:
        raise NotImplementedError

    def __call__(self, bat):
        raise NotImplementedError

    def get_loss(self, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        return self.generator.state_dict()

    def parameters(self):
        return self.generator.parameters()

    def load_state_dict(self, state_dict):
        self.generator.load_state_dict(state_dict)

    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, **kwargs):
        raise NotImplementedError