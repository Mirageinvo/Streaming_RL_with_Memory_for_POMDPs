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

def preprocess_obs_batch(obs_batch: np.ndarray, device: torch.device):
    x = torch.from_numpy(obs_batch).to(device)
    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    return x

def preprocess_obs_single(obs: np.ndarray, device: torch.device):
    x = torch.from_numpy(np.asarray(obs)).to(device)
    if x.dtype != torch.float32:
        x = x.float()
    x = x.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    return x

def save_logs_npz(path: str, returns, episode_steps, episodes, term_time_steps, compress: bool = False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = dict(
        returns=np.asarray(returns, dtype=np.float32),
        episode_steps=np.asarray(episode_steps, dtype=np.int64),
        episodes=np.asarray(episodes, dtype=np.int64),
        term_time_steps=np.asarray(term_time_steps, dtype=np.int64),
    )
    np.savez(path, **data)


class RecurrentActorCriticCNN(nn.Module):
    def __init__(self, obs_shape, n_actions: int, hidden: int = 256, lstm_size: int = 256):
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

        self.lstm = nn.LSTMCell(hidden, lstm_size)

        self.policy = nn.Linear(lstm_size, n_actions)
        self.value = nn.Linear(lstm_size, 1)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def init_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(batch_size, self.lstm.hidden_size, device=device)
        return h, c

    def forward_step(self, x: torch.Tensor, state):
        h, c = state
        z = self.conv(x)
        z = self.fc(z)
        h, c = self.lstm(z, (h, c))
        logits = self.policy(h)
        v = self.value(h).squeeze(-1)
        return logits, v, (h, c)

    @torch.no_grad()
    def act_step(self, x: torch.Tensor, state):
        logits, v, next_state = self.forward_step(x, state)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent, v, next_state

def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = ImgObsWrapper(env)
    obs, _ = env.reset(seed=seed)
    return env, obs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="MiniGrid-MemoryS7-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--total_steps", type=int, default=500_000)
    ap.add_argument("--n_steps", type=int, default=2048, help="rollout length per update (single env)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--seq_len", type=int, default=32, help="BPTT sequence length")
    ap.add_argument("--seqs_per_batch", type=int, default=8, help="how many sequences per minibatch")
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lstm_size", type=int, default=256)
    ap.add_argument("--log_path", type=str, default="", help="Optional .npz log path")
    ap.add_argument("--log_compress", action="store_true")
    ap.add_argument("--save_path", type=str, default="", help="Optional model .pt path")
    ap.add_argument("--print_every", type=int, default=10, help="print every N updates")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    env, obs = make_env(args.env, args.seed)
    obs_shape = obs.shape

    action_map = None
    n_actions = env.action_space.n

    model = RecurrentActorCriticCNN(obs_shape, n_actions=n_actions, hidden=args.hidden, lstm_size=args.lstm_size).to(device)
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
    seq_len = args.seq_len
    assert seq_len > 0 and seq_len <= n_steps

    n_updates = int(np.ceil(args.total_steps / n_steps))

    obs_t = obs
    h_t, c_t = model.init_state(batch_size=1, device=device)

    for update in range(1, n_updates + 1):
        obs_buf = np.zeros((n_steps,) + obs_shape, dtype=np.uint8)
        actions_buf = np.zeros((n_steps,), dtype=np.int64)
        logp_buf = np.zeros((n_steps,), dtype=np.float32)
        rewards_buf = np.zeros((n_steps,), dtype=np.float32)
        dones_buf = np.zeros((n_steps,), dtype=np.float32)
        values_buf = np.zeros((n_steps,), dtype=np.float32)

        h_buf = np.zeros((n_steps, args.lstm_size), dtype=np.float32)
        c_buf = np.zeros((n_steps, args.lstm_size), dtype=np.float32)

        for t in range(n_steps):
            obs_buf[t] = obs_t
            h_buf[t] = h_t.squeeze(0).detach().cpu().numpy()
            c_buf[t] = c_t.squeeze(0).detach().cpu().numpy()

            x = preprocess_obs_single(obs_t, device)  # (1,C,H,W)
            a, logp, _, v, (h_next, c_next) = model.act_step(x, (h_t, c_t))

            a_idx = int(a.item())
            env_a = action_map[a_idx] if action_map is not None else a_idx

            actions_buf[t] = a_idx
            logp_buf[t] = float(logp.item())
            values_buf[t] = float(v.item())

            next_obs, r, terminated, truncated, _ = env.step(env_a)
            done = bool(terminated or truncated)

            rewards_buf[t] = float(r)
            dones_buf[t] = 1.0 if done else 0.0

            ep_ret += float(r)
            ep_len += 1
            global_step += 1

            h_t, c_t = h_next, c_next
            if done:
                episode_returns.append(ep_ret)
                episode_steps.append(ep_len)
                episodes.append(ep_idx)
                term_time_steps.append(global_step)

                ep_idx += 1
                ep_ret = 0.0
                ep_len = 0

                next_obs, _ = env.reset(seed=args.seed + ep_idx)
                h_t, c_t = model.init_state(batch_size=1, device=device)

            obs_t = next_obs

        with torch.no_grad():
            x_last = preprocess_obs_single(obs_t, device)
            _, last_v, _ = model.forward_step(x_last, (h_t, c_t))
            last_value = float(last_v.item())

        advantages = np.zeros((n_steps,), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values_buf[t + 1]
            next_nonterminal = 1.0 - dones_buf[t]
            delta = rewards_buf[t] + args.gamma * next_value * next_nonterminal - values_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = preprocess_obs_batch(obs_buf, device)
        actions_tensor = torch.from_numpy(actions_buf).to(device)
        old_logp_tensor = torch.from_numpy(logp_buf).to(device)
        returns_tensor = torch.from_numpy(returns).to(device)
        adv_tensor = torch.from_numpy(advantages).to(device)
        dones_tensor = torch.from_numpy(dones_buf).to(device)

        h_tensor = torch.from_numpy(h_buf).to(device)
        c_tensor = torch.from_numpy(c_buf).to(device)

        n_seq = n_steps // seq_len
        assert n_seq != 0
        starts = np.arange(0, n_seq * seq_len, seq_len, dtype=np.int64)

        seqs_per_batch = args.seqs_per_batch
        assert seqs_per_batch > 0

        idx_arange = torch.arange(seq_len, device=device).unsqueeze(0)

        for _epoch in range(args.n_epochs):
            np.random.shuffle(starts)
            for mb_start in range(0, len(starts), seqs_per_batch):
                mb_starts = starts[mb_start: mb_start + seqs_per_batch]
                if len(mb_starts) == 0:
                    continue

                mb_starts_t = torch.from_numpy(mb_starts).to(device)
                idx = mb_starts_t.unsqueeze(1) + idx_arange

                mb_obs = obs_tensor[idx]
                mb_actions = actions_tensor[idx]
                mb_old_logp = old_logp_tensor[idx]
                mb_returns = returns_tensor[idx]
                mb_adv = adv_tensor[idx]
                mb_dones = dones_tensor[idx]

                h0 = h_tensor[mb_starts_t]
                c0 = c_tensor[mb_starts_t]

                h = h0
                c = c0

                new_logps = []
                entropies = []
                values = []

                for t in range(seq_len):
                    x_t = mb_obs[:, t]
                    logits, v_t, (h, c) = model.forward_step(x_t, (h, c))

                    dist = Categorical(logits=logits)
                    a_t = mb_actions[:, t]
                    new_logps.append(dist.log_prob(a_t))
                    entropies.append(dist.entropy())
                    values.append(v_t)

                    done_t = mb_dones[:, t].unsqueeze(1)
                    h = h * (1.0 - done_t)
                    c = c * (1.0 - done_t)

                new_logp = torch.stack(new_logps, dim=1)
                entropy = torch.stack(entropies, dim=1)
                value = torch.stack(values, dim=1)

                new_logp_f = new_logp.reshape(-1)
                old_logp_f = mb_old_logp.reshape(-1)
                adv_f = mb_adv.reshape(-1)
                ret_f = mb_returns.reshape(-1)
                value_f = value.reshape(-1)
                entropy_f = entropy.reshape(-1)

                ratio = torch.exp(new_logp_f - old_logp_f)
                surr1 = ratio * adv_f
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * adv_f
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (ret_f - value_f).pow(2).mean()
                entropy_loss = entropy_f.mean()

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
