"""
Evaluate a saved DQN model on the GridWorldEnv.

Produces a CSV with columns:
  run,episode,steps,total_reward,success

Usage example:
  python evaluate_model.py --model model/DQN_2025-11-01.12:00:00.pth --out results.csv --runs 5 --episodes 100
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import torch

# reuse definitions from solve.py
from solve import DQN, obs_to_state, device
from RL.Env.Environment import GridWorldEnv

from utils.DirectoryManager import DirectoryManager
from LTL_tasks import formulas

def print_sample_execution(env: GridWorldEnv, model: DQN, max_steps: int):
    obs, _, _ = env.reset()
    state_tensor = obs_to_state(obs, env)
    total_reward = 0.0
    done = False
    truncated = False
    info = {}
    steps = 0

    for t in range(max_steps):
        steps += 1
        with torch.no_grad():
            qvals = model(state_tensor)
            # handle batched or single-sample outputs
            if qvals.dim() == 2:
                action = int(qvals.argmax(dim=1)[0].cpu().item())
            else:
                action = int(qvals.argmax().cpu().item())

        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        past_state = state_tensor
        state_tensor = obs_to_state(next_obs, env)
        print(f"Step {steps}: state = {past_state}, action={action}, next_state={state_tensor}, reward={reward:.2f}, total_reward={total_reward:.4f}, info={info}")
        if done or truncated:
            print("Done:", done, "Truncated:", truncated)
            break

    print(f"Sample execution: steps={steps}, total_reward={total_reward:.2f}, success={done}")

def evaluate(model_path: str, out_csv: str, runs: int, episodes: int, max_steps: int,
             env_kwargs: dict):
    model_path = Path(model_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # build environment
    env = GridWorldEnv(**env_kwargs)

    # construct model matching env and load weights
    model = DQN(env).to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    with out_csv.open("w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["run", "episode", "steps", "total_reward", "success"])

        for run in range(1, runs + 1):
            for ep in range(1, episodes + 1):
                obs, _, _ = env.reset()
                state_tensor = obs_to_state(obs, env)
                total_reward = 0.0
                done = False
                truncated = False
                info = {}
                steps = 0

                for t in range(max_steps):
                    steps += 1
                    with torch.no_grad():
                        qvals = model(state_tensor)
                        # handle batched or single-sample outputs
                        if qvals.dim() == 2:
                            action = int(qvals.argmax(dim=1)[0].cpu().item())
                        else:
                            action = int(qvals.argmax().cpu().item())

                    next_obs, reward, done, truncated, info = env.step(action)
                    total_reward += float(reward)
                    state_tensor = obs_to_state(next_obs, env)
                    if done or truncated:
                        break

                writer.writerow([run, ep, steps, f"{total_reward:.4f}", 1 if done else 0])

    # compute simple summary
    data = np.genfromtxt(str(out_csv), delimiter=",", names=True, dtype=None, encoding=None)
    rewards = np.array([float(x["total_reward"]) for x in data])
    successes = np.array([int(x["success"]) for x in data])
    steps_array = np.array([int(x["steps"]) for x in data])
    print(f"Saved per-episode results to {out_csv}")
    print(f"Overall episodes: {len(rewards)} | steps/episode {steps_array.mean():.2f} | mean_reward: {rewards.mean():.2f} | success_rate: {successes.mean():.2f}")

    print_sample_execution(env, model, max_steps)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("timestamp", help="timestamp of the experiment to evaluate")
    p.add_argument("--model", help="path to saved model .pth")
    p.add_argument("--runs", type=int, default=1, help="number of independent runs")
    p.add_argument("--episodes", type=int, default= 10, help="episodes per run")
    p.add_argument("--max_steps", type=int, default=50, help="max steps per episode")
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--state_type", choices=["symbolic", "image"], default="symbolic")
    p.add_argument("--use_dfa", action="store_true", default=True)
    args = p.parse_args()

    timestamp = args.timestamp
    dm = DirectoryManager(timestamp)
    for formula_name in dm.get_formula_names():
        print(f"Found formula: {formula_name}")
        for formula in formulas:
            if formula[2].replace(" ", "_") == formula_name:
                env_kwargs = dict(
                    formula=formula,
                    render_mode="rgb_array",
                    state_type=args.state_type,
                    use_dfa_state=args.use_dfa,
                    train=False,
                    size=args.size,
                )
                dm.set_formula_name(formula_name)
                for model_path in dm.get_models():
                    print(f"Evaluating model: {model_path}")
                    out = f"data/{timestamp}/{formula_name}/eval/eval_{model_path.stem}.csv"
                    evaluate(str(model_path), out, args.runs, args.episodes, args.max_steps, env_kwargs)