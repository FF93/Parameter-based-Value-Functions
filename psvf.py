import torch
import numpy as np
import gym
import core
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions.normal import Normal

# Default hyperparameters
hyperparameter_defaults = dict(
    algo='psvf',
    env_name='Swimmer-v3',
    neurons_policy=(),
    neurons_vf=(512,512),
    policy_iters=1,
    vf_iters=5,
    batch_size=128,
    learning_rate_policy=1e-3,
    learning_rate_vf=1e-4,
    noise_policy=1.0,  # std of distribution generating the noise for the perturbed policy
    observation_normalization=True,
    size_buffer=100000,
    max_episodes=1000000000,
    max_timesteps=1000000,
    run=1,
    deterministic_actor=True,
    ts_evaluation=10000,
    update_every_ts=50,
    discount_factor=0.999,
    bs_policy_update=10000,
)

# Initialize wandb
wandb.init(config=hyperparameter_defaults, project="psvf_rl")
config = wandb.config

# Use GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# Create env
env = gym.make(config['env_name'])
env_test = gym.make(config['env_name'])

# Create replay buffer, policy, vf
buffer = core.Buffer_td(config['size_buffer'])
statistics = core.Statistics(env.observation_space.shape)
ac = core.MLPActorCritic(config['algo'], env.observation_space, env.action_space,
                         hidden_sizes_actor=tuple(config['neurons_policy']), activation=nn.Tanh,
                         hidden_sizes_critic=tuple(config['neurons_vf']), device=device,
                         critic=True, deterministic_actor=config['deterministic_actor'])

print("number of policy params:", len(nn.utils.parameters_to_vector(list(ac.pi.parameters()))))
print("number of vf params:", len(nn.utils.parameters_to_vector(list(ac.v.parameters()))))

# Setup optimizer
optimize_policy = optim.Adam(ac.pi.parameters(), lr=config['learning_rate_policy'])
optimize_vf = optim.Adam(ac.v.parameters(), lr=config['learning_rate_vf'])

wandb.watch(ac.pi)
wandb.watch(ac.v)


def compute_policy_loss(parameters, states):
    parameters = parameters.repeat(len(states), 1)
    losses = -ac.v.forward(parameters, states)
    loss = torch.mean(losses)
    return loss


def compute_vf_loss(states, parameters, next_states, rewards, is_terminal):
    with torch.no_grad():
        next_estimate = ac.v.forward(parameters, next_states)
    current_estimate = ac.v.forward(parameters, states)

    loss = ((current_estimate - (rewards + config['discount_factor'] * (1 - is_terminal) * next_estimate))**2).mean()

    return loss


def perturbe_policy(policy):

    dist = Normal(torch.zeros(len(torch.nn.utils.parameters_to_vector(policy.parameters()))), scale=1)
    delta = dist.sample().to(device, non_blocking=True).detach()

    # Perturbe policy parameters
    params = torch.nn.utils.parameters_to_vector(policy.parameters()).detach()
    perturbed_params = params + config['noise_policy'] * delta

    # Copy perturbed parameters into a new policy
    perturbed_policy = core.MLPActorCritic(config['algo'], env.observation_space, env.action_space,
                                           hidden_sizes_actor=tuple(config['neurons_policy']), activation=nn.Tanh,
                                           hidden_sizes_critic=tuple(config['neurons_vf']), device=device,
                                           critic=False, deterministic_actor=config['deterministic_actor'])

    torch.nn.utils.vector_to_parameters(perturbed_params, perturbed_policy.parameters())

    return perturbed_policy


def update():

    for i in range(config['vf_iters']):
        # Sample batch
        sampled_hist = buffer.sample_replay_td(config['batch_size'])
        sampled_programs, sampled_transitions = zip(*sampled_hist)
        sampled_states, sampled_rewards, sampled_next_states, sampled_terminal_condition = zip(*sampled_transitions)
        sampled_programs = torch.stack(sampled_programs).to(device, non_blocking=True).detach()
        sampled_states = torch.stack(sampled_states).to(device, non_blocking=True).detach()

        sampled_next_states = torch.stack(sampled_next_states).to(device, non_blocking=True).detach()
        sampled_rewards = torch.stack(sampled_rewards).to(device, non_blocking=True).detach()
        sampled_terminal_condition = torch.stack(sampled_terminal_condition).to(device, non_blocking=True).detach()

        optimize_vf.zero_grad()
        loss_vf = compute_vf_loss(sampled_states, sampled_programs, sampled_next_states, sampled_rewards, sampled_terminal_condition)
        loss_vf.backward()
        optimize_vf.step()

    # Freeze PSVF
    for p in ac.v.parameters():
        p.requires_grad = False

    # Update policy
    sampled_hist_up = buffer.sample_replay_td(config['bs_policy_update'])
    _, sampled_transitions_up = zip(*sampled_hist_up)
    sampled_states_up, _, _, _ = zip(*sampled_transitions_up)
    sampled_states_up = torch.stack(sampled_states_up).to(device, non_blocking=True).detach()

    for i in range(config['policy_iters']):
        params = nn.utils.parameters_to_vector(list(ac.pi.parameters())).to(device, non_blocking=True)

        optimize_policy.zero_grad()
        loss_policy = compute_policy_loss(params, sampled_states_up)
        loss_policy.backward()
        optimize_policy.step()

    # Unfreeze PSVF
    for p in ac.v.parameters():
        p.requires_grad = True
    return


def evaluate(ac):
    rew_evals = []
    with torch.no_grad():
        for _ in range(10):

            # Simulate a trajectory and compute the total reward
            done = False
            obs = env_test.reset()
            rew_eval = 0
            while not done:
                obs = torch.as_tensor(obs, dtype=torch.float32)
                if config['observation_normalization'] and statistics.episode > 0:
                    obs = statistics.normalize(obs)

                with torch.no_grad():
                    action = ac.act(obs.to(device, non_blocking=True).detach())
                obs_new, r, done, _ = env_test.step(action)

                # Remove survival bonus
                if config['env_name'] == 'Hopper-v3':
                    rew_eval += r - 1
                else:
                    rew_eval += r
                obs = obs_new

            rew_evals.append(rew_eval)

        statistics.rew_eval = np.mean(rew_evals)
        statistics.push_rew(np.mean(rew_evals))
    # Log results

    wandb.log({'rew_eval': statistics.rew_eval,
               'average_reward': np.mean(statistics.rewards),
               'average_last_rewards': np.mean(statistics.last_rewards),
               })
    print("Ts", statistics.total_ts, "Ep", statistics.episode, "rew_eval", statistics.rew_eval)

    return


def train():
    obs = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32)
    if config['observation_normalization']:
        statistics.push_obs(obs)
        obs = statistics.normalize(obs)

    rew = 0
    # Perturbe policy
    perturbed_policy = perturbe_policy(ac.pi)
    perturbed_params = nn.utils.parameters_to_vector(list(perturbed_policy.parameters())).to(device, non_blocking=True).detach()

    while statistics.total_ts < config['max_timesteps'] and statistics.episode < config['max_episodes']:

        # Collect data
        with torch.no_grad():
            action = perturbed_policy.act(obs.to(device, non_blocking=True).detach())
        obs_new, r, done, _ = env.step(action)

        # Remove survival bonus
        if config['env_name'] == 'Hopper-v3':
            rew += r - 1
        else:
            rew += r

        statistics.total_ts += 1
        statistics.len_episode += 1
        obs_new = torch.as_tensor(obs_new, dtype=torch.float32)
        if config['observation_normalization']:
            statistics.push_obs(obs_new)
            obs_new = statistics.normalize(obs_new)

        done_bool = float(done) if statistics.len_episode < env._max_episode_steps else 0

        transition = (obs, torch.tensor(r).float(), obs_new, torch.tensor(float(done_bool)))
        buffer.push((perturbed_params.detach(), transition))

        obs = obs_new

        if done:
            # Log results
            wandb.log({'rew': rew,
                       'steps': statistics.total_ts,
                       'episode': statistics.episode,
                       })

            obs = env.reset()
            obs = torch.as_tensor(obs, dtype=torch.float32)
            if config['observation_normalization']:
                statistics.push_obs(obs)
                obs = statistics.normalize(obs)
            rew = 0
            statistics.episode += 1

            # Perturbe policy
            perturbed_policy = perturbe_policy(ac.pi)
            perturbed_params = nn.utils.parameters_to_vector(list(perturbed_policy.parameters())).to(device,
                                                                                                     non_blocking=True).detach()

            statistics.len_episode = 0

        # Update
        if statistics.total_ts % config['update_every_ts'] == 0:
            update()

        # Evaluate current policy
        if statistics.total_ts % config['ts_evaluation'] == 0:
            evaluate(ac)

    return


# Initial evaluation
evaluate(ac)

# Loop over episodes
train()

