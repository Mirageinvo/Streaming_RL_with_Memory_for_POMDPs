import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from torch.distributions import Categorical


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_obs(obs_batch: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs_batch).to(device)
    if x.dtype != torch.float32:
        x = x.float()
    assert x.ndim == 4
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


class ActorCriticCNN(nn.Module):
    def __init__(self, obs_shape, n_actions: int, hidden: int = 256):
        super().__init__()
        assert len(obs_shape) == 3
        H, W, C = obs_shape

        if C not in (1, 3) and obs_shape[0] in (1, 3):
            C, H, W = obs_shape[0], obs_shape[1], obs_shape[2]

        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            y = self.conv(dummy)
            conv_dim = int(y.numel())

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dim, hidden),
            nn.ReLU(),
        )

        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
        logits = self.policy(h)
        v = self.value(h).squeeze(-1)
        return logits, v

    @torch.no_grad()
    def get_action_value(self, x):
        logits, v = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent, v

    def evaluate_actions(self, x, actions):
        logits, v = self.forward(x)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ent = dist.entropy()
        return logp, ent, v


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = ImgObsWrapper(env)
    obs, _ = env.reset(seed=seed)
    return env, obs

def save_logs_npz(path: str, returns, episode_steps, episodes, term_time_steps, compress: bool = False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = dict(
        returns=np.asarray(returns, dtype=np.float32),
        episode_steps=np.asarray(episode_steps, dtype=np.int64),
        episodes=np.asarray(episodes, dtype=np.int64),
        term_time_steps=np.asarray(term_time_steps, dtype=np.int64),
    )
    np.savez(path, **data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="MiniGrid-MemoryS7-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--total_steps", type=int, default=500_000)
    ap.add_argument("--n_steps", type=int, default=2048, help="rollout length per update")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--log_path", type=str, default="")
    ap.add_argument("--log_compress", action="store_true")
    ap.add_argument("--save_path", type=str, default="")
    ap.add_argument("--print_every", type=int, default=10)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    env, obs = make_env(args.env, args.seed)
    n_actions = env.action_space.n
    obs_shape = obs.shape

    model = ActorCriticCNN(obs_shape, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    episode_returns = []
    episode_steps = []
    episodes = []
    term_time_steps = []
    global_step = 0
    ep_idx = 0
    ep_ret = 0.0
    ep_len = 0

    n_steps = args.n_steps
    batch_size = args.batch_size
    n_updates = int(np.ceil(args.total_steps / n_steps))

    obs_t = obs

    for update in range(1, n_updates + 1):
        obs_buf = np.zeros((n_steps,) + obs_shape, dtype=np.uint8)
        actions_buf = np.zeros((n_steps,), dtype=np.int64)
        logp_buf = np.zeros((n_steps,), dtype=np.float32)
        rewards_buf = np.zeros((n_steps,), dtype=np.float32)
        dones_buf = np.zeros((n_steps,), dtype=np.float32)
        values_buf = np.zeros((n_steps,), dtype=np.float32)

        for t in range(n_steps):
            obs_buf[t] = obs_t

            x = preprocess_obs(obs_t[None, ...], device)
            with torch.no_grad():
                a, logp, _, v = model.get_action_value(x)

            a_item = int(a.item())
            actions_buf[t] = a_item
            logp_buf[t] = float(logp.item())
            values_buf[t] = float(v.item())

            next_obs, r, terminated, truncated, _ = env.step(a_item)
            done = bool(terminated or truncated)

            rewards_buf[t] = float(r)
            dones_buf[t] = 1.0 if done else 0.0

            ep_ret += float(r)
            ep_len += 1
            global_step += 1

            if done:
                episode_returns.append(ep_ret)
                episode_steps.append(ep_len)
                episodes.append(ep_idx)
                term_time_steps.append(global_step)

                ep_idx += 1
                ep_ret = 0.0
                ep_len = 0
                next_obs, _ = env.reset(seed=args.seed + ep_idx)

            obs_t = next_obs

        with torch.no_grad():
            x_last = preprocess_obs(obs_t[None, ...], device)
            _, last_value = model(x_last)
            last_value = float(last_value.item())

        advantages = np.zeros((n_steps,), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_nonterminal = 1.0 - dones_buf[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones_buf[t]
                next_value = values_buf[t + 1]

            delta = rewards_buf[t] + args.gamma * next_value * next_nonterminal - values_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_buf
        adv_mean = float(advantages.mean())
        adv_std = float(advantages.std() + 1e-8)
        advantages = (advantages - adv_mean) / adv_std

        obs_tensor = preprocess_obs(obs_buf, device)
        actions_tensor = torch.from_numpy(actions_buf).to(device)
        old_logp_tensor = torch.from_numpy(logp_buf).to(device)
        returns_tensor = torch.from_numpy(returns).to(device)
        adv_tensor = torch.from_numpy(advantages).to(device)

        inds = np.arange(n_steps)
        for _epoch in range(args.n_epochs):
            np.random.shuffle(inds)
            for start in range(0, n_steps, batch_size):
                mb_inds = inds[start : start + batch_size]
                mb_obs = obs_tensor[mb_inds]
                mb_actions = actions_tensor[mb_inds]
                mb_old_logp = old_logp_tensor[mb_inds]
                mb_returns = returns_tensor[mb_inds]
                mb_adv = adv_tensor[mb_inds]

                new_logp, entropy, values = model.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % args.print_every == 0:
            if len(episode_returns) > 0:
                last_k = min(50, len(episode_returns))
                avg_ret = float(np.mean(episode_returns[-last_k:]))
                print(f"Update {update:4d}/{n_updates} | steps={global_step} | avg_return(last{last_k})={avg_ret:.3f}")
            else:
                print(f"Update {update:4d}/{n_updates} | steps={global_step} | (no completed episodes yet)")

        if global_step >= args.total_steps:
            break

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved model to: {args.save_path}")

    if args.log_path:
        save_logs_npz(
            args.log_path,
            returns=episode_returns,
            episode_steps=episode_steps,
            episodes=episodes,
            term_time_steps=term_time_steps,
            compress=args.log_compress,
        )
        print(f"Saved logs to: {args.log_path}")


if __name__ == "__main__":
    main()
