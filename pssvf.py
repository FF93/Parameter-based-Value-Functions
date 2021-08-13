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
    algo='pssvf',
    env_name='CartPole-v1',
    neurons_policy=(),
    neurons_vf=(64,64),
    policy_iters=10,
    vf_iters=10,
    batch_size=16,
    learning_rate_policy=1e-3,
    learning_rate_vf=1e-3,
    noise_policy=1.0, # std of distribution generating the noise for the perturbed policy
    observation_normalization=True,
    size_buffer=100000,
    max_episodes=1000000000,
    max_timesteps=100000,
    run=1,
    deterministic_actor=True,
    ts_evaluation=1000,
)

# Initialize wandb
wandb.init(config=hyperparameter_defaults, project="pssvf_rl")
config = wandb.config

# Use GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# Create env
env = gym.make(config['env_name'])
env_test = gym.make(config['env_name'])

# Create replay buffer, policy, vf
buffer = core.Buffer(config['size_buffer'])
statistics = core.Statistics(env.observation_space.shape)
ac = core.MLPActorCritic(config['algo'], env.observation_space, env.action_space,
                         hidden_sizes_actor=tuple(config['neurons_policy']), activation=nn.Tanh,
                         hidden_sizes_critic=tuple(config['neurons_vf']), device=device,
                         critic=True, deterministic_actor=config['deterministic_actor'])

print("Number of policy params:", len(nn.utils.parameters_to_vector(list(ac.pi.parameters()))))
print("Number of value function params:", len(nn.utils.parameters_to_vector(list(ac.v.parameters()))))

# Setup optimizer
optimize_policy = optim.Adam(ac.pi.parameters(), lr=config['learning_rate_policy'])
optimize_vf = optim.Adam(ac.v.parameters(), lr=config['learning_rate_vf'])

wandb.watch(ac.pi)
wandb.watch(ac.v)


def compute_policy_loss(parameters):
    return -ac.v.forward(parameters)


def compute_vf_loss(parameters, rewards):
    return ((ac.v(parameters) - rewards)**2).mean()


def perturbe_policy(policy):

    dist = Normal(torch.zeros(len(torch.nn.utils.parameters_to_vector(policy.parameters()))), scale=1)
    delta = dist.sample().to(device=device, non_blocking=True).detach()

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

    for _ in range(config['vf_iters']):
        # Sample batch
        hist = buffer.sample_replay(config['batch_size'])
        prog, rew = zip(*hist)
        prog = torch.stack(prog)
        rew = torch.from_numpy(np.asarray(rew)).float().to(device=device, non_blocking=True).detach()

        optimize_vf.zero_grad()
        loss_vf = compute_vf_loss(prog, rew)
        loss_vf.backward()
        optimize_vf.step()

    # Freeze PSSVF
    for p in ac.v.parameters():
        p.requires_grad = False

    # Update policy
    for _ in range(config['policy_iters']):
        params = nn.utils.parameters_to_vector(list(ac.pi.parameters())).to(device, non_blocking=True)

        optimize_policy.zero_grad()
        loss_policy = compute_policy_loss(params)
        loss_policy.backward()
        optimize_policy.step()

    # Unfreeze PSSVF
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


def simulate(ac):

    # Simulate a trajectory and compute the total reward
    done = False
    obs = env.reset()
    rew = 0
    while not done:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if config['observation_normalization']:
            statistics.push_obs(obs)
            if statistics.episode > 0:
                obs = statistics.normalize(obs)

        with torch.no_grad():
            action = ac.act(obs.to(device, non_blocking=True).detach())
        obs_new, r, done, _ = env.step(action)
        # Remove survival bonus
        if config['env_name'] == 'Hopper-v3':
            rew += r - 1
        else:
            rew += r

        statistics.total_ts += 1

        # Evaluate current policy
        if statistics.total_ts % config['ts_evaluation'] == 0:
            evaluate(ac)

        obs = obs_new

    return rew


def train():

    # Collect data with perturbed policy
    perturbed_policy = perturbe_policy(ac.pi)
    # Extract list of perturbed policy parameters
    perturbed_params = nn.utils.parameters_to_vector(list(perturbed_policy.parameters())).to(device, non_blocking=True).detach()
    # Simulate a trajectory and compute the total reward
    rew = simulate(perturbed_policy)
    # Store data in replay buffer
    buffer.history.append((perturbed_params, rew))
    statistics.episode += 1
    # Update
    update()
    # Log results
    wandb.log({'rew': rew,
               'steps': statistics.total_ts,
               'episode': statistics.episode,
               })

    return


# Initial evaluation
evaluate(ac)

# Loop over episodes
while statistics.total_ts < config['max_timesteps'] and statistics.episode < config['max_episodes']:
    train()
