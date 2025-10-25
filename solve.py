## Solve Environment.
import random
import collections
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Assumes Environment.GridWorldEnv is importable from RL.Env.Environment
from RL.Env.Environment import GridWorldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, np.array(actions), np.array(rewards, dtype=np.float32), next_states, np.array(dones, dtype=np.uint8)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, env: GridWorldEnv, hidden=128):
        super().__init__()
        self.state_type = env.state_type
        self.use_dfa = env.use_dfa_state

        if self.state_type == "image":
            # small CNN to extract features from 3x64x64 images
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            # compute cnn output size with a dummy pass
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 64, 64)
                cnn_out = self.cnn(dummy).shape[1]
            if self.use_dfa:
                dfa_dim = env.automaton.num_of_states
            else:
                dfa_dim = 0
            self.fc = nn.Sequential(
                nn.Linear(cnn_out + dfa_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, env.action_space.n),
            )
        else:
            # symbolic: observation is a vector (e.g. x,y,dfa_state) or (x,y) + dfa
            obs_dim = env.state_space_size if isinstance(env.state_space_size, int) else int(np.prod(env.state_space_size))
            if self.use_dfa:
                obs_dim += env.automaton.num_of_states
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, env.action_space.n),
            )

    def forward(self, state):
        # state can be:
        #  - symbolic: single tensor vector
        #  - image: tuple (one_hot_dfa_tensor, image_tensor) or just image_tensor if not using dfa
        if self.state_type == "image":
            if isinstance(state, tuple) or isinstance(state, list):
                dfa, img = state
                img = img.to(device).float().unsqueeze(0)
                cnn_feat = self.cnn(img)
                if dfa is not None:
                    dfa = dfa.to(device).float().unsqueeze(0)
                    x = torch.cat([cnn_feat, dfa], dim=1)
                else:
                    x = cnn_feat
            else:
                # just image tensor
                img = state.to(device).float().unsqueeze(0)
                cnn_feat = self.cnn(img)
                x = cnn_feat
            return self.fc(x)
        else:
            x = state.to(device).float().unsqueeze(0)
            return self.fc(x)

def obs_to_state(obs, env: GridWorldEnv):
    # normalize/convert observation to tensors in the shape expected by DQN.forward
    if env.state_type == "symbolic":
        # If using DFA state we expect obs to contain the symbolic state and a one-hot/dfa part,
        # or a single concatenated vector. Ensure the return has shape (obs_dim,)
        if env.use_dfa_state:
            # obs might be (state_vec, one_hot) or already concatenated array
            if isinstance(obs, (tuple, list)) and len(obs) == 2:
                state_vec, one_hot = obs
                state_t = torch.tensor(np.array(state_vec).astype(np.float32))
                one_hot_t = torch.tensor(np.array(one_hot).astype(np.float32))
                return torch.cat([state_t, one_hot_t])
            else:
                arr = np.array(obs).astype(np.float32)
                return torch.from_numpy(arr)
        else:
            # simple numeric/vector observation
            if isinstance(obs, np.ndarray):
                return torch.from_numpy(obs.astype(np.float32))
            else:
                return torch.tensor(np.array(obs).astype(np.float32))
    else:
        # image mode: reset/step return either image only or [one_hot, image_tensor]
        if env.use_dfa_state:
            one_hot = obs[0]
            img = obs[1]
            one_hot_t = torch.tensor(np.array(one_hot).astype(np.float32))
            if not torch.is_tensor(img):
                img_t = torch.tensor(np.array(img).astype(np.float32))
            else:
                img_t = img.float()
            return (one_hot_t, img_t)
        else:
            img = obs
            if not torch.is_tensor(img):
                img_t = torch.tensor(np.array(img).astype(np.float32))
            else:
                img_t = img.float()
            return img_t


def train(env: GridWorldEnv, episodes=1000, batch_size=64, gamma=0.99, lr=1e-4,
          buffer_capacity=20000, target_update=1000, start_train=1000, max_steps_per_episode=200):

    online = DQN(env).to(device)
    target = DQN(env).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    steps_done = 0
    eps_start, eps_end, eps_decay = 1.0, 0.05, 30000

    for ep in range(1, episodes + 1):
        obs, _, _ = env.reset()
        ## Check obs and state format
        state = obs_to_state(obs, env)
        print(obs, obs.__class__, state, state.__class__)  ## Print for debugging
        total_reward = 0.0
        done = False
        truncated = False

        for t in range(max_steps_per_episode):
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
            steps_done += 1

            # select action
            if random.random() < eps_threshold:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    print(obs, obs.__class__) ## Print for debugging
                    qvals = online(state)
                    action = int(torch.argmax(qvals, dim=1).cpu().numpy()[0])

            next_obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            next_state = obs_to_state(next_obs, env)
            terminal = bool(done or truncated)
            buffer.push(state, action, reward, next_state, terminal)

            state = next_state

            # optimize
            if len(buffer) >= max(64, start_train):
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)

                # prepare tensors
                def stack_states(s_list):
                    # convert list of states (some are tuples for images) into batched tensors for forward
                    if env.state_type == "image":
                        if env.use_dfa_state:
                            dfa_batch = torch.stack([s[0] for s in s_list]).to(device)
                            img_batch = torch.stack([s[1] for s in s_list]).to(device)
                            return (dfa_batch, img_batch)
                        else:
                            return torch.stack([s for s in s_list]).to(device)
                    else:
                        return torch.stack([s for s in s_list]).to(device)

                states_t = stack_states(states_b)
                next_states_t = stack_states(next_states_b)
                actions_t = torch.tensor(actions_b, device=device, dtype=torch.long).unsqueeze(1)
                rewards_t = torch.tensor(rewards_b, device=device, dtype=torch.float32).unsqueeze(1)
                dones_t = torch.tensor(dones_b, device=device, dtype=torch.float32).unsqueeze(1)

                q_values = online(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = target(next_states_t)
                    next_q_max = next_q.max(1)[0].unsqueeze(1)
                    target_q = rewards_t + (1.0 - dones_t) * gamma * next_q_max

                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % target_update == 0:
                target.load_state_dict(online.state_dict())

            if terminal:
                break

        # simple logging
        if ep % 10 == 0:
            print(f"Episode {ep:4d} | steps {steps_done:6d} | reward {total_reward:.2f} | epsilon {eps_threshold:.3f} | buffer {len(buffer)}")

    # save model
    torch.save(online.state_dict(), "dqn_gridworld.pth")
    print("Training finished. Model saved to dqn_gridworld.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--formula", nargs=3, default=["(F c0) & (F c1)", "2", "task0: visit(gem, door)"])
    parser.add_argument("--state_type", choices=["symbolic", "image"], default="symbolic")
    parser.add_argument("--use_dfa", action="store_true", default=True)
    args = parser.parse_args()

    # Example: adapt the formula argument to the repository's expected format.
    # Here formula = (ltl_formula_str, num_symbols(int/string), formula_name)
    env = GridWorldEnv(formula=(args.formula[0], int(args.formula[1]), args.formula[2]),
                       render_mode="rgb_array", state_type=args.state_type, use_dfa_state=args.use_dfa, train=True, size=args.size)
    train(env, episodes=args.episodes)