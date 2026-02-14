import os
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from torch.distributions import Categorical

warnings.filterwarnings("ignore", category=DeprecationWarning)

from optim import ObGD as Optimizer
from sparse_init import sparse_init


def initialize_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        sparse_init(m.weight, sparsity=0.9)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class StreamACMiniGridLSTM(nn.Module):
    def __init__(
        self,
        obs_shape,
        n_actions: int,
        hidden_size: int = 256,
        lstm_size: int = 256,
        lr_head: float = 1.0,
        lr_mem: float = 3e-4,
        gamma: float = 0.99,
        lamda: float = 0.8,
        kappa_policy: float = 3.0,
        kappa_value: float = 2.0,
        tbptt_steps: int = 20,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.tbptt_steps = int(tbptt_steps)
        self.max_grad_norm = float(max_grad_norm)
        self.device = torch.device(device)

        assert len(obs_shape) == 3

        H, W, C = obs_shape
        if C not in (1, 3) and obs_shape[0] in (1, 3):
            C = obs_shape[0]
            H = obs_shape[1]
            W = obs_shape[2]
        self._H, self._W, self._C = int(H), int(W), int(C)

        self.conv1 = nn.Conv2d(self._C, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            dummy = torch.zeros(1, self._C, self._H, self._W)
            y = F.relu(self.conv1(dummy))
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y))
            conv_dim = int(y.numel())

        self.enc_fc = nn.Linear(conv_dim, hidden_size)

        self.lstm = nn.LSTMCell(hidden_size, lstm_size)

        self.policy_head = nn.Linear(lstm_size, n_actions)
        self.value_head = nn.Linear(lstm_size, 1)

        self.apply(initialize_weights)

        self.optimizer_policy = Optimizer(
            self.policy_head.parameters(),
            lr=lr_head,
            gamma=self.gamma,
            lamda=lamda,
            kappa=kappa_policy,
        )
        self.optimizer_value = Optimizer(
            self.value_head.parameters(),
            lr=lr_head,
            gamma=self.gamma,
            lamda=lamda,
            kappa=kappa_value,
        )

        mem_params = (
            list(self.conv1.parameters())
            + list(self.conv2.parameters())
            + list(self.conv3.parameters())
            + list(self.enc_fc.parameters())
            + list(self.lstm.parameters())
        )
        self.optim_mem = torch.optim.Adam(mem_params, lr=lr_mem)

        self._mem_losses = []
        self._steps_since_tbptt = 0

        self.to(self.device)
        self.reset_state()


    def reset_state(self):
        hsz = self.lstm.hidden_size
        self.h = torch.zeros(1, hsz, device=self.device)
        self.c = torch.zeros(1, hsz, device=self.device)
        self._mem_losses = []
        self._steps_since_tbptt = 0

    def _prep_obs(self, obs):
        x = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).unsqueeze(0)

        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()

        return x

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "conv1": self.conv1.state_dict(),
                "conv2": self.conv2.state_dict(),
                "conv3": self.conv3.state_dict(),
                "enc_fc": self.enc_fc.state_dict(),
                "lstm": self.lstm.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
            },
            path,
        )

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.conv1.load_state_dict(ckpt["conv1"])
        self.conv2.load_state_dict(ckpt["conv2"])
        self.conv3.load_state_dict(ckpt["conv3"])
        self.enc_fc.load_state_dict(ckpt["enc_fc"])
        self.lstm.load_state_dict(ckpt["lstm"])
        self.policy_head.load_state_dict(ckpt["policy_head"])
        self.value_head.load_state_dict(ckpt["value_head"])


    def _encode(self, x_nchw):
        y = F.relu(self.conv1(x_nchw))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.flatten(1)
        z = F.relu(self.enc_fc(y))
        return z

    def forward_step(self, obs):
        x = self._prep_obs(obs)
        z = self._encode(x)
        self.h, self.c = self.lstm(z, (self.h, self.c))
        logits = self.policy_head(self.h)
        value = self.value_head(self.h).squeeze(-1)
        return logits, value, (self.h, self.c)

    @torch.no_grad()
    def act(self, obs):
        logits, _, _ = self.forward_step(obs)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        a = dist.sample().item()
        return a

    def sample_action_from_logits(self, logits):
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        a = dist.sample()
        return a.item(), dist

    def _accumulate_mem_loss(self, h, a_tensor, td_target, entropy_coeff: float):
        logits = F.linear(h, self.policy_head.weight.detach(), self.policy_head.bias.detach())
        value = F.linear(h, self.value_head.weight.detach(), self.value_head.bias.detach()).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        adv = (td_target - value).detach()
        logp = dist.log_prob(a_tensor)
        entropy = dist.entropy()

        policy_loss = -(logp * adv)
        value_loss = 0.5 * (td_target.detach() - value).pow(2)
        ent_bonus = -float(entropy_coeff) * entropy

        loss = (policy_loss + value_loss + ent_bonus).mean()
        self._mem_losses.append(loss)

    def _tbptt_update_mem(self, reset: bool):
        if len(self._mem_losses) == 0:
            if reset:
                self.h = self.h.detach()
                self.c = self.c.detach()
            return

        if (self._steps_since_tbptt >= self.tbptt_steps) or reset:
            self.optim_mem.zero_grad()
            loss = torch.stack(self._mem_losses).sum()
            loss.backward()
            mem_params = (
                list(self.conv1.parameters())
                + list(self.conv2.parameters())
                + list(self.conv3.parameters())
                + list(self.enc_fc.parameters())
                + list(self.lstm.parameters())
            )
            torch.nn.utils.clip_grad_norm_(mem_params, self.max_grad_norm)
            self.optim_mem.step()

            self.h = self.h.detach()
            self.c = self.c.detach()
            self._mem_losses = []
            self._steps_since_tbptt = 0


    def update_heads_streamx(self, h_detached, a_t: int, r_t: float, v_next_detached, done: bool, entropy_coeff: float):
        done_mask = 0.0 if done else 1.0
        r = torch.tensor(float(r_t), device=self.device)
        done_mask = torch.tensor(done_mask, device=self.device)

        logits_s = self.policy_head(h_detached)
        v_s = self.value_head(h_detached).squeeze(-1)
        v_prime = v_next_detached

        td_target = r + self.gamma * v_prime * done_mask
        delta = (td_target - v_s).detach()

        probs = F.softmax(logits_s, dim=-1)
        dist = Categorical(probs)

        a_tensor = torch.tensor(a_t, device=self.device)
        log_prob_pi = -(dist.log_prob(a_tensor)).sum()

        value_output = -v_s.mean()
        entropy_pi = -float(entropy_coeff) * dist.entropy().sum() * torch.sign(delta).item()

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()

        value_output.backward()
        (log_prob_pi + entropy_pi).backward()

        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        return td_target

    def train_step(self, obs_t, a_t: int, r_t: float, obs_tp1, done: bool, entropy_coeff: float = 0.01):
        h_t_det = self.h.detach()

        h_saved, c_saved = self.h, self.c
        logits_tp1, v_tp1, _ = self.forward_step(obs_tp1)
        v_next_det = v_tp1.detach()
        self.h, self.c = h_saved, c_saved

        td_target = self.update_heads_streamx(h_t_det, a_t, r_t, v_next_det, done, entropy_coeff)

        a_tensor = torch.tensor(a_t, device=self.device)
        self._accumulate_mem_loss(self.h, a_tensor, td_target, entropy_coeff)

        self._steps_since_tbptt += 1
        self._tbptt_update_mem(reset=done)

        if done:
            self.reset_state()

def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = ImgObsWrapper(env)
    obs, _ = env.reset(seed=seed)
    return env, obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-MemoryS7-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tbptt", type=int, default=80)
    parser.add_argument("--lr_mem", type=float, default=3e-4)
    parser.add_argument("--lr_head", type=float, default=1.0)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--log_path", type=str, default="")
    parser.add_argument("--log_compress", action="store_true")
    args = parser.parse_args()

    env, obs = make_env(args.env, args.seed)
    n_actions = env.action_space.n

    agent = StreamACMiniGridLSTM(
        obs_shape=obs.shape,
        n_actions=n_actions,
        device=args.device,
        tbptt_steps=args.tbptt,
        lr_mem=args.lr_mem,
        lr_head=args.lr_head,
    )

    episode_returns = []
    episode_steps = []
    episodes = []
    term_time_steps = []
    global_step = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        agent.reset_state()
        done = False
        ep_ret = 0.0
        ep_len = 0

        for _ in range(args.max_steps):
            logits, _, _ = agent.forward_step(obs)
            a, _ = agent.sample_action_from_logits(logits)

            next_obs, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)

            agent.train_step(obs, a, float(r), next_obs, done, entropy_coeff=args.entropy)

            obs = next_obs
            ep_ret += float(r)
            ep_len += 1
            global_step += 1

            if done:
                break

        episode_returns.append(ep_ret)
        episode_steps.append(ep_len)
        episodes.append(ep)
        term_time_steps.append(global_step)

        if (ep + 1) % 50 == 0:
            avg = float(np.mean(episode_returns[-50:]))
            print(f"Episode {ep+1:5d} | avg_return(last50)={avg:.3f}")

    if args.save_path:
        agent.save(args.save_path)
        print(f"Saved model to: {args.save_path}")

    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
        data = dict(
            returns=np.asarray(episode_returns, dtype=np.float32),
            episode_steps=np.asarray(episode_steps, dtype=np.int64),
            episodes=np.asarray(episodes, dtype=np.int64),
            term_time_steps=np.asarray(term_time_steps, dtype=np.int64),
        )
        np.savez(args.log_path, **data)
        print(f"Saved logs to: {args.log_path}")


if __name__ == "__main__":
    main()
