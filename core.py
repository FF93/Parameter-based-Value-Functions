import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete
import numpy as np
import random


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]

    return nn.Sequential(*layers)


class MLPStochasticCategoricalActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation)

    def _distribution(self, obs):
        logits = self.pi_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs):
        pi = self._distribution(obs)
        a = pi.sample()
        return a


class MLPDeterministicCategoricalActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation)
        self.softmax = nn.Softmax()

    def forward(self, obs):
        logits = self.pi_net(obs)
        out = self.softmax(logits)
        a = out.argmax()
        return a


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act_limit = act_limit

    def _distribution(self, obs):
        mu = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs):

        pi = self._distribution(obs)
        a = torch.tanh(pi.sample())
        return (self.act_limit * a)


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit
    def forward(self, obs):
        return (self.act_limit * self.pi_net(obs))



class PSSVF(nn.Module):
    def __init__(self, parameter_space_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([parameter_space_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, parameters):
        return torch.squeeze(self.v_net(parameters), -1)


class PSVF(nn.Module):

    def __init__(self, parameter_space_dim, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([parameter_space_dim + obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, parameters, observations):
        h = torch.cat((parameters, observations), dim=-1)
        return torch.squeeze(self.v_net(h), -1)


class PAVF(nn.Module):

    def __init__(self, parameter_space_dim, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([parameter_space_dim + obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, parameters, observations, actions):
        h = torch.cat((parameters, observations, actions), dim=-1)
        return torch.squeeze(self.v_net(h), -1)


class MLPActorCritic(nn.Module):

    def __init__(self, algo, observation_space, action_space,
                 hidden_sizes_actor, activation, hidden_sizes_critic, device, critic,
                 deterministic_actor):
        super().__init__()

        self.algo = algo

        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
            act_limit = action_space.high[0]
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n

        if isinstance(action_space, Box):
            if deterministic_actor:
                self.pi = MLPActor(obs_dim, act_dim, hidden_sizes_actor,
                                   activation, act_limit).to(device=device)
            else:
                self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes_actor,
                                           activation, act_limit).to(device=device)

        elif isinstance(action_space, Discrete):
            if deterministic_actor:
                self.pi = MLPDeterministicCategoricalActor(obs_dim, action_space.n,
                                                           hidden_sizes_actor, activation).to(device=device)
            else:
                self.pi = MLPStochasticCategoricalActor(obs_dim, action_space.n,
                                                        hidden_sizes_actor, activation).to(device=device)

        if critic:
            self.parameters_dim = len(nn.utils.parameters_to_vector(list(self.pi.parameters())))

            if self.algo == 'pssvf':
                self.v = PSSVF(self.parameters_dim, hidden_sizes_critic, nn.ReLU).to(device=device)
            elif self.algo == 'psvf':
                self.v = PSVF(self.parameters_dim, obs_dim, hidden_sizes_critic, nn.ReLU).to(device=device)
            elif self.algo == 'pavf':
                self.v = PAVF(self.parameters_dim, obs_dim, act_dim, hidden_sizes_critic, nn.ReLU).to(device=device)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).to(device='cpu').numpy()


class Statistics(object):

    def __init__(self, obs_dim):
        super().__init__()

        self.total_ts = 0
        self.episode = 0
        self.len_episode = 0
        self.rew_shaped_eval = 0
        self.rew_eval = 0
        self.rewards = []
        self.last_rewards = []
        self.position = 0
        self.n = 0
        self.mean = torch.zeros(obs_dim)
        self.mean_diff = torch.zeros(obs_dim)
        self.std = torch.zeros(obs_dim)


    def push_obs(self, obs):
        self.n += 1.
        last_mean = self.mean
        self.mean += (obs - self.mean) / self.n
        self.mean_diff += (obs - last_mean) * (obs - self.mean)
        var = self.mean_diff / (self.n - 1) if self.n > 1 else np.square(self.mean)
        self.std = np.sqrt(var)
        return

    def push_rew(self, rew):
        if len(self.last_rewards) < 20:
            self.last_rewards.append(rew)
        else:
            self.last_rewards[self.position] = rew
            self.position = (self.position + 1) % 20
        self.rewards.append(rew)

    def normalize(self, obs):
        return (obs - self.mean) / (self.std + 1e-8)


class Buffer(object):
    def __init__(self, size_buffer):
        self.history = []
        self.size_buffer = size_buffer

    def sample_replay(self, batch_size):

        sampled_hist = random.sample(self.history, min(int(batch_size), len(self.history)))
        if len(self.history) > self.size_buffer:
            self.history.pop(0)
        return sampled_hist


class Buffer_td(object):
    def __init__(self, capacity):
        self.history = []
        self.capacity = capacity
        self.position = 0

    def push(self, transition):
        if len(self.history) < self.capacity:
            self.history.append(transition)
        else:
            self.history[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample_replay_td(self, batch_size):

        sampled_trans = random.choices(self.history, k=int(batch_size))
        return sampled_trans
