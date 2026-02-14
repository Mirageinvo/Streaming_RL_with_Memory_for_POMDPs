import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
import torch.nn.functional as F
from torch.distributions import Categorical
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from optim import ObGD as Optimizer
from sparse_init import sparse_init

class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return F.layer_norm(input, input.size())
    def extra_repr(self) -> str:
        return "Layer Normalization"

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class StreamACMiniGrid(nn.Module):
    def __init__(self, n_actions=3, hidden_size=256, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamACMiniGrid, self).__init__()
        self.gamma = gamma
        self.network_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(147, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.network_policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(147, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        self.apply(initialize_weights)
        self.optimizer_policy = Optimizer(self.network_policy.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.network_value.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float).unsqueeze(0)
        preferences = self.network_policy(x).squeeze(0)
        probs = F.softmax(preferences, dim=-1)
        return probs

    def v(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float).unsqueeze(0)
        return self.network_value(x).squeeze(0)

    def sample_action(self, s):
        probs = self.pi(s)
        dist = Categorical(probs)
        return dist.sample().item()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float64), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float64), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float64)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        probs = self.pi(s)
        dist = Categorical(probs)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

    def save(self, path):
        torch.save({
            'policy_state_dict': self.network_policy.state_dict(),
            'value_state_dict': self.network_value.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network_policy.load_state_dict(checkpoint['policy_state_dict'])
        self.network_value.load_state_dict(checkpoint['value_state_dict'])

def main(env_name, seed, lr, gamma, lamda, total_steps, entropy_coeff, kappa_policy, kappa_value, debug, overshooting_info, save_model, load_path, render=False):
    if not load_path:
        torch.manual_seed(seed); np.random.seed(seed)
        env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
        env = ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        agent = StreamACMiniGrid(n_actions=env.action_space.n, lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
        if debug:
            print("seed: {}".format(seed), "env: {}".format(env.spec.id))
        returns, term_time_steps = [], []
        step_list, reward_list = [], []
        s, _ = env.reset(seed=seed)
        episode_num = 1
        for t in range(1, total_steps+1):
            a = agent.sample_action(s)
            s_prime, r, terminated, _, info = env.step(a)
            agent.update_params(s, a, r, s_prime, terminated, entropy_coeff, overshooting_info)
            s = s_prime
            step_list.append(t)
            reward_list.append(r)
            if info and "episode" in info:
                if debug:
                    print("Episodic Return: {}, Time Step {}, Episode Number {}".format(info['episode']['r'][0], t, episode_num))
                returns.append(info['episode']['r'][0])
                term_time_steps.append(t)
                s, _ = env.reset()
                episode_num += 1
        env.close()
        np.savez_compressed(
            f"logs_{env_name}_seed{seed}.npz",
            step=np.asarray(step_list, dtype=np.int64),
            reward=np.asarray(reward_list, dtype=np.float32),
            returns=np.asarray(returns, dtype=np.float32),
            term_time_steps=np.asarray(term_time_steps, dtype=np.int64),
        )
        save_dir = "data_stream_ac_{}_lr{}_gamma{}_lamda{}_entropy_coeff{}".format(env.spec.id, lr, gamma, lamda, entropy_coeff)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
            pickle.dump((returns, term_time_steps, env_name), f)
        if save_model:
            agent.save(os.path.join(save_dir, f"model_seed_{seed}.pt"))
    else:
        torch.manual_seed(seed); np.random.seed(seed)
        env = gym.make(env_name, render_mode='human')
        env = ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        agent = StreamACMiniGrid(n_actions=env.action_space.n, lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
        agent.load(load_path)
        agent.eval()
        if debug:
            print("seed: {}".format(seed), "env: {}".format(env.spec.id))
        returns = []
        s, _ = env.reset(seed=seed)
        episode_num = 1
        with torch.no_grad():
            for t in range(1, total_steps+1):
                probs = agent.pi(s)
                a = torch.argmax(probs).item()
                s_prime, r, terminated, _, info = env.step(a)
                s = s_prime
                if info and "episode" in info:
                    if debug:
                        print("Episodic Return: {}, Time Step {}, Episode Number {}".format(info['episode']['r'][0], t, episode_num))
                    returns.append(info['episode']['r'][0])
                    s, _ = env.reset()
                    episode_num += 1
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default="MiniGrid-MemoryS7-v0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=200_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.entropy_coeff, args.kappa_policy, args.kappa_value, args.debug, args.overshooting_info, args.save_model, args.load_path, args.render)