import math
import logging

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiplicativeLR, ReduceLROnPlateau
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup

from options import NEG_METRICS

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_optimizer_class(name):

    if name == "Adam":
        return torch.optim.Adam
    elif name == "RAdam":
        return RAdam
    elif name == "AdamW":
        return AdamW


def get_optimizers(args, model):

    optim_class = get_optimizer_class(args.optimizer)

    no_decay_str = ['bias', 'LayerNorm.weight']
    encoder_no_decay_pgroup = []  # parameter group corresponding to encoder, contains biases and LayerNorm params
    encoder_decay_pgroup = []  # parameter group corresponding to encoder. L2 regularization will be applied
    nonencoder_no_decay_pgroup = []  # parameter group for non-encoder part, contains biases and LayerNorm params
    nonencoder_decay_pgroup = []  # parameter group for non-encoder part. L2 regularization will be applied
    for name, param in model.named_parameters():
        if name.startswith('encoder'):
            if any(st in name for st in no_decay_str):
                encoder_no_decay_pgroup.append(param)
            else:
                encoder_decay_pgroup.append(param)
        else:
            if any(st in name for st in no_decay_str):
                nonencoder_no_decay_pgroup.append(param)
            else:
                nonencoder_decay_pgroup.append(param)

    encoder_optim_pgroups = [{'params': encoder_no_decay_pgroup, 'weight_decay': 0},
                             {'params': encoder_decay_pgroup, 'weight_decay': args.weight_decay}]
    # keyword arguments here will be the global defaults
    encoder_optimizer = optim_class(encoder_optim_pgroups, lr=args.encoder_learning_rate, eps=args.adam_epsilon)

    nonencoder_optim_pgroups = [{'params': nonencoder_no_decay_pgroup, 'weight_decay': 0},
                                {'params': nonencoder_decay_pgroup, 'weight_decay': args.weight_decay}]
    # keyword arguments here will be the global defaults
    nonencoder_optimizer = optim_class(nonencoder_optim_pgroups, lr=args.learning_rate, eps=args.adam_epsilon)

    return nonencoder_optimizer, encoder_optimizer


def get_schedulers(args, total_training_steps, nonencoder_optimizer, encoder_optimizer):

    if args.reduce_on_plateau is not None:
        mode = 'min' if args.reduce_on_plateau in NEG_METRICS else 'max'
        patience = round(args.ROP_patience / args.validation_steps)  # patience in number of evaluations
        ROP_nonencoder_scheduler = ReduceLROnPlateau(nonencoder_optimizer, mode=mode, factor=args.ROP_factor,
                                                     patience=patience,
                                                     cooldown=args.ROP_cooldown,
                                                     verbose=True,
                                                     threshold_mode=args.ROP_thr_mode,
                                                     threshold=args.ROP_threshold,
                                                     min_lr=args.final_lr_ratio*args.learning_rate)
        ROP_encoder_scheduler = ReduceLROnPlateau(encoder_optimizer, mode=mode, factor=args.ROP_factor,
                                                  patience=patience,
                                                  cooldown=args.ROP_cooldown,
                                                  verbose=True,
                                                  threshold_mode=args.ROP_thr_mode,
                                                  threshold=args.ROP_threshold,
                                                  min_lr=args.final_lr_ratio*args.encoder_learning_rate)

        nonencoder_scheduler = get_constant_schedule_with_warmup(nonencoder_optimizer, num_warmup_steps=args.warmup_steps)
        encoder_scheduler = get_constant_schedule_with_warmup(encoder_optimizer, num_warmup_steps=args.encoder_warmup_steps)

        return {'nonencoder_scheduler': nonencoder_scheduler, 'encoder_scheduler': encoder_scheduler,
                'ROP_nonencoder_scheduler': ROP_nonencoder_scheduler, 'ROP_encoder_scheduler': ROP_encoder_scheduler}

    nonencoder_scheduler = get_polynomial_decay_schedule_with_warmup(nonencoder_optimizer,
                                                                     num_warmup_steps=args.warmup_steps,
                                                                     num_training_steps=total_training_steps,
                                                                     lr_end=args.final_lr_ratio * args.learning_rate,
                                                                     power=1.0)
    encoder_scheduler = get_polynomial_decay_schedule_with_warmup(encoder_optimizer,
                                                                  num_warmup_steps=args.encoder_warmup_steps,
                                                                  num_training_steps=total_training_steps,
                                                                  lr_end=args.final_lr_ratio*args.encoder_learning_rate,
                                                                  power=1.0)

    return {'nonencoder_scheduler': nonencoder_scheduler, 'encoder_scheduler': encoder_scheduler}


class MultiOptimizer(object):
    """Provides a single-Optimizer API for a collection of several optimizers.
    Useful in case several schedules are used to control the learning rate of different portions of the model."""

    def __init__(self, *optimizers):
        """Ex: multoptim = MultiOptimizer(optim1, optim2)"""
        self.optimizers = list(optimizers)

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        num_states = len(self.optimizers)
        if len(state_dicts) > len(self.optimizers):
            logger.warning("Got {} state dictionaries for {} optimizers! "
                           "Will only use the first {} state dictionaries.".format(len(state_dicts), len(self.optimizers), len(self.optimizers)))
        elif len(state_dicts) < len(self.optimizers):
            logger.warning("Got {} state dictionaries for {} optimizers! "
                           "The state of some optimizers will remain as is.".format(len(state_dicts), len(self.optimizers)))
            num_states = len(state_dicts)
        for i in range(num_states):
            self.optimizers[i].load_state_dict(state_dicts[i])

    def add_optimizer(self, optimizer):
        self.optimizers.append(optimizer)

    def pop_optimizer(self):
        self.optimizers.pop()


class MultiScheduler(object):
    """Provides a single-Scheduler API for a collection of several schedulers.
    Useful in case several schedules are used to control the learning rate of different portions of the model."""

    def __init__(self, *schedulers):
        """Ex: multischeduler = MultiScheduler(sched1, sched2)"""
        self.schedulers = list(schedulers)

    def step(self, *args):
        for s in self.schedulers:
            s.step(*args)

    def state_dict(self):
        return [s.state_dict() for s in self.schedulers]

    def load_state_dict(self, state_dicts):
        num_states = len(self.schedulers)
        if len(state_dicts) > len(self.schedulers):
            logger.warning("Got {} state dictionaries for {} schedulers! "
                           "Will only use the first {} state dictionaries.".format(len(state_dicts), len(self.schedulers), len(self.schedulers)))
        elif len(state_dicts) < len(self.schedulers):
            logger.warning("Got {} state dictionaries for {} schedulers! "
                           "The state of some schedulers will remain as is.".format(len(state_dicts), len(self.schedulers)))
            num_states = len(state_dicts)
        for i in range(num_states):
            self.schedulers[i].load_state_dict(state_dicts[i])

    def get_last_lr(self):
        return [s.get_last_lr() for s in self.schedulers]

    def add_scheduler(self, scheduler):
        self.schedulers.append(scheduler)

    def pop_scheduler(self):
        self.schedulers.pop()


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period, during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    The returned scheduler can be combined with other schedulers, such as ReduceOnPlateau.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.MultiplicativeLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        lr_init = 1/(num_warmup_steps - 1)
        if current_step == 1:
            return lr_init
        if current_step < num_warmup_steps:
            return float(current_step) / float(current_step - 1)
        return 1.0

    scheduler = MultiplicativeLR(optimizer, lr_lambda, last_epoch=last_epoch)

    return scheduler


# from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW2(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamW2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW2, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
