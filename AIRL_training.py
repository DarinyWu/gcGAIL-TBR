# -*- coding: utf-8 -*-
"""
AIRL benchmark pipeline with:
- Hyperparameter random search
- Convergence logging & plots
- Running time recording/report
- Reward signal reporting (expert vs generator)
- Test-set metrics (accuracy, confusion matrix)

Author: Yuanyuan Wu & ChatGPT
"""

import os
import time
import json
import math
import random
import pathlib
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout as il_rollout
from imitation.testing.reward_improvement import is_significant_reward_improvement

from sklearn.metrics import confusion_matrix, classification_report

# ===========
# GLOBAL DEFAULTS (overridable by CLI)
# ===========
SEED = 42
# prepare your data###
EXPERT_TRAIN_PATH = 'Dateset_train.csv'
TEST_PATH = 'Dataset_test.csv'

TOTAL_TIMESTEPS = 1_638_400
EVAL_INTERVAL = 16_384           # used for *logging cadence* (in rounds below)
EARLY_PATIENCE = 20
DISC_ACC_TOL = 0.05              # early stop if in [0.45, 0.55]

PRETRAIN_PPO = True
PRETRAIN_TIMESTEPS = 300_000

RUN_HYPERPARAM_SEARCH = True
N_TRIALS_RANDOM_SEARCH = 12
SEARCH_BUDGET_STEPS = 200_000    # per trial; rounded down to full rounds

# Outputs
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = pathlib.Path(f"outputs/AIRL_{TS}")
PLOTS_DIR = OUT_DIR / "plots"
LOGS_DIR = OUT_DIR / "logs"
MODELS_DIR = OUT_DIR / "models"
REPORTS_DIR = OUT_DIR / "reports"
for d in [OUT_DIR, PLOTS_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===========
# UTILS
# ===========
def set_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def save_json(obj, path):
    """Save dict as JSON, converting NumPy and Path types to built-ins."""
    import numpy as np
    import pathlib

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pathlib.Path,)):
            return str(o)
        return str(o)  # fallback

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def plot_series(x, ys: Dict[str, List[float]], title: str, out_path: pathlib.Path, xlabel="step", ylabel="value"):
    plt.figure(figsize=(8,5))
    for k, v in ys.items():
        if len(x) == len(v):
            plt.plot(x, v, label=k)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def hist_plot(data_dict: Dict[str, np.ndarray], title: str, out_path: pathlib.Path, bins: int = 50):
    plt.figure(figsize=(8,5))
    for name, arr in data_dict.items():
        plt.hist(arr, bins=bins, alpha=0.5, density=True, label=name)
    plt.title(title)
    plt.xlabel("reward")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ===========
# DATA
# ===========
def load_expert_data(file_path: str) -> Dict[Any, List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]]:
    """Load, normalize, one-hot, and pack trajectories grouped by PassengerID."""
    df = pd.read_csv(file_path)
    expert_data = {}

    normalization_ranges = {
        'Months_Passed': (-1, 13),
        'LastMonthMode': (0, 1),
        'TwoMonthMode': (0, 1),
        'LastTripStart': (-113, 160),
        'LastTripEnd': (-73, 164),
        'LastTripFare': (0, 80),
        'LastMinShift': (0, 104),
        'MeanMorningPeakMoneySave': (0, 12),
        'FirstWeekdayMorningTapInTimeStd': (0.11, 150),
        'AvgTripTravelTime': (-2.4, 68),
        'MeanTimeShift': (0, 104),
    }

    home_loc_dim = 121
    work_loc_dim = 121
    work_loc_is_disc_sta_dim = 2

    for expert_id, group in df.groupby('PassengerID'):
        states = group[
            ['HomeLoc', 'WorkLoc', 'WorkLocIsDiscSta', 'Months_Passed', 'LastMonthMode', 'TwoMonthMode',
             'LastTripStart',
             'LastTripEnd', 'LastTripFare', 'LastMinShift',
             'MeanMorningPeakMoneySave', 'FirstWeekdayMorningTapInTimeStd', 'AvgTripTravelTime',
             'MeanTimeShift']].values.astype(np.float32)

        # Convert first 3 columns to integer for one-hot encoding
        home_loc = states[:, 0].astype(int)
        work_loc = states[:, 1].astype(int)
        work_loc_is_disc_sta = states[:, 2].astype(int)

        # One-hot encode 'HomeLoc', 'WorkLoc', 'WorkLocIsDiscSta'
        home_loc_one_hot = np.eye(home_loc_dim)[home_loc]  # One-hot encode HomeLoc
        work_loc_one_hot = np.eye(work_loc_dim)[work_loc]  # One-hot encode WorkLoc
        work_loc_is_disc_sta_one_hot = np.eye(work_loc_is_disc_sta_dim)[work_loc_is_disc_sta]  # One-hot encode binary

        # Normalize the rest (columns 3 and onwards) to [0,1]
        for i, col_name in enumerate(normalization_ranges.keys()):
            min_val, max_val = normalization_ranges[col_name]
            states[:, i + 3] = (states[:, i + 3] - min_val) / (max_val - min_val)  # Min-max normalization
            states[:, i + 3] = np.clip(states[:, i + 3], 0, 1)  # Ensure values stay in [0, 1]

        # Concatenate one-hot encoded values with normalized states
        states = np.concatenate([home_loc_one_hot, work_loc_one_hot, work_loc_is_disc_sta_one_hot, states[:, 3:]],
                                axis=1)

        actions = group[['MonthMode']].values

        next_states = np.roll(states, -1, axis=0)  # shift obs to get next_obs
        next_states[-1] = np.zeros(states.shape[1])  # Set last next_obs to zero to avoid incorrect shift

        # Extract rewards and dones
        rewards = group['LastTripFare'].values
        dones = group['done'].values.astype(bool)  # Convert to boolean array
        dones = dones.tolist()  # Convert NumPy array to a list of booleans

        # Create a list of (state, action, next_state, reward) tuples for each expert
        trajectory = list(zip(states, actions, next_states, rewards, dones))
        expert_data[expert_id] = trajectory

    return expert_data

def convert_to_transitions(expert_data: Dict[Any, Any], add_noise: bool = False) -> Transitions:
    all_obs, all_acts, all_next_obs, all_rewards, all_dones = [], [], [], [], []
    for _, trajectory in expert_data.items():
        for obs, act, next_obs, reward, done in trajectory:
            all_obs.append(obs)
            all_acts.append(act[0])
            all_next_obs.append(next_obs)
            all_rewards.append(reward)
            all_dones.append(done)
    all_obs = np.array(all_obs, dtype=np.float32)
    all_acts = np.array(all_acts, dtype=np.float32)
    all_next_obs = np.array(all_next_obs, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    all_dones = np.array(all_dones, dtype=bool)
    all_infos = [{}] * len(all_obs)
    if add_noise:
        noise = np.random.normal(0, 0.01, all_obs.shape)
        all_obs += noise
    return Transitions(obs=all_obs, acts=all_acts, next_obs=all_next_obs, dones=all_dones, infos=all_infos)

# ===========
# ENV
# ===========

class CustomEnv(gym.Env):
    """A custom environment for gcGAIL."""

    def __init__(self, expert_data):
        super(CustomEnv, self).__init__()
        self.expert_data = expert_data
        self.expert_ids = list(expert_data.keys())
        self.num_experts = len(self.expert_ids)

        # Define action space (1-dimensional continuous)
        #self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # Define observation space
        #self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float64)
        self.num_categorical = [121, 121,2]  # home, work, location_if, Three categorical features with numbers of categories
        self.continuous_dim = 11 # Number of continuous variables

        # Calculate flattened observation space size
        self.flat_observation_size = (
                self.continuous_dim  # Continuous features remain the same
                + sum(self.num_categorical)  # One-hot encoding for categorical variables
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.flat_observation_size,), dtype=np.float64)
        self.current_trajectory = None
        self.current_step = 0
        self.max_steps = 15

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed)
        kwargs.pop('options', None)  # Remove 'options' if it exists
        expert_id = np.random.choice(self.expert_ids)
        self.current_trajectory = self.expert_data[expert_id]
        self.current_step = 0
        obs = self.current_trajectory[self.current_step][0]
        return obs, {}

    def step(self, action):
        """Take an action and return the next state, reward, done, and info."""
        state, true_action, next_state, reward, done = self.current_trajectory[self.current_step]

        # Calculate imitation reward: reward based on distance between action and expert's action
        imitation_reward = 1.0 if true_action == action else 0.0

        self.current_step += 1
        if not done:
            obs = next_state
        else:
            obs, _ = self.reset()
            imitation_reward = 0
            done = False

        return obs, imitation_reward, done, False, {}

    def render(self, mode='human'):
        # Optionally implemented rendering logic
        pass

    def close(self):
        pass


# ===========
# BUILDERS
# ===========
def make_vec_env(expert_data, n_envs=8, max_ep_steps=15, seed=SEED):
    def _make_env():
        _env = CustomEnv(expert_data)
        _env = TimeLimit(_env, max_episode_steps=max_ep_steps)
        _env = RolloutInfoWrapper(_env)
        return _env
    venv = DummyVecEnv([_make_env for _ in range(n_envs)])
    venv.seed(seed)
    # Wrap with VecMonitor so evaluation sees clean episode stats
    venv = VecMonitor(venv, filename=str(LOGS_DIR / "monitor"))
    return venv

def make_reward_net(venv):
    return BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

def build_ppo(env, p: Dict[str, Any]) -> PPO:
    return PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=p.get("batch_size", 256),
        ent_coef=p.get("ent_coef", 0.0),
        learning_rate=p.get("learning_rate", 1e-4),
        gamma=p.get("gamma", 0.95),
        n_epochs=p.get("n_epochs", 30),
        n_steps=p.get("n_steps", 2048),
        gae_lambda=p.get("gae_lambda", 0.95),
        clip_range=p.get("clip_range", 0.2),
        seed=SEED,
        device=p.get("device", "cpu"),
        policy_kwargs=p.get("policy_kwargs", dict(net_arch=[256, 256])),
        tensorboard_log=str(LOGS_DIR / "tb"),
        verbose=0,
    )

# ===========
# HYPERPARAM SEARCH (round-safe)
# ===========
def sample_random_params() -> Dict[str, Any]:
    return {
        "learning_rate": 10 ** np.random.uniform(-5.5, -3.5),
        "gamma": np.random.choice([0.95, 0.98, 0.99]),
        "batch_size": np.random.choice([128, 256, 512]),
        "n_steps": np.random.choice([1024, 2048]),  # keep tidy for gen_batch_size
        "n_epochs": np.random.choice([10, 20, 30]),
        "gae_lambda": np.random.choice([0.90, 0.95, 0.99]),
        "clip_range": np.random.choice([0.1, 0.2, 0.3]),
        "ent_coef": np.random.choice([0.0, 0.001, 0.01]),
        "policy_kwargs": dict(net_arch=list(np.random.choice([128, 256, 384], size=2)))
    }

def quick_train_eval(expert_transitions, expert_data, params, budget_steps=200_000) -> float:
    venv = make_vec_env(expert_data)
    learner = build_ppo(venv, params)
    reward_net = make_reward_net(venv)
    airl = AIRL(
        demonstrations=expert_transitions,
        demo_batch_size=256,
        gen_replay_buffer_capacity=128,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    gen_batch_size = learner.n_steps * venv.num_envs
    # round the budget to full rounds
    budget_steps = max(gen_batch_size, budget_steps - (budget_steps % gen_batch_size))
    n_rounds = budget_steps // gen_batch_size

    for _ in range(n_rounds):
        airl.train(gen_batch_size)

    rewards, _ = evaluate_policy(learner, venv, n_eval_episodes=10, return_episode_rewards=True)
    mean_r = float(np.mean(rewards))
    try: venv.close()
    except: pass
    return mean_r

def hyperparam_search(expert_transitions, expert_data) -> Dict[str, Any]:
    results = []
    best = (-math.inf, None)
    t0 = time.perf_counter()
    for i in range(N_TRIALS_RANDOM_SEARCH):
        params = sample_random_params()
        mean_r = quick_train_eval(expert_transitions, expert_data, params, budget_steps=SEARCH_BUDGET_STEPS)
        results.append({**params, "mean_reward": mean_r})
        if mean_r > best[0]:
            best = (mean_r, params)
        print(f"[Search] Trial {i+1}/{N_TRIALS_RANDOM_SEARCH}: meanR={mean_r:.3f}, params={params}")
    elapsed = time.perf_counter() - t0
    pd.DataFrame(results).to_csv(REPORTS_DIR / "hyperparam_search_results.csv", index=False)

    def to_json_safe(x):
        if isinstance(x, dict):
            return {k: to_json_safe(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_json_safe(v) for v in x]
        import numpy as _np
        if isinstance(x, (_np.integer,)): return int(x)
        if isinstance(x, (_np.floating,)): return float(x)
        if isinstance(x, _np.ndarray): return x.tolist()
        return x

    best_mean, best_params = best
    payload = {"best_mean_reward": float(best_mean),
               "best_params": to_json_safe(best_params),
               "search_seconds": float(elapsed)}
    save_json(payload, REPORTS_DIR / "hyperparam_search_summary.json")

    return best[1] if best[1] is not None else {}

# ===========
# MAIN
# ===========
def main():
    set_seeds(SEED)
    runtime = {}

    # Data
    t0 = time.perf_counter()
    expert_data = load_expert_data(EXPERT_TRAIN_PATH)
    expert_transitions = convert_to_transitions(expert_data)
    runtime["load_data_sec"] = time.perf_counter() - t0

    # Initial PPO defaults (can be overridden by search)
    tuning_params = {
        "learning_rate": 1e-4,
        "gamma": 0.95,
        "batch_size": 256,
        "n_epochs": 30,
        "n_steps": 2048,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "policy_kwargs": dict(net_arch=[256, 256]),
    }
    # tuning_params = {
    #     "learning_rate": 2.546151773535996e-05,
    #     "gamma": 0.98,
    #     "batch_size": 256,
    #     "n_steps": 2048,
    #     "n_epochs": 20,
    #     "gae_lambda": 0.90,
    #     "clip_range": 0.30,
    #     "ent_coef": 0.0,
    #     "policy_kwargs": {"net_arch": [128, 256]},
    # }

    # Hyperparam search
    if RUN_HYPERPARAM_SEARCH:
        print(">>> Running random-search hyperparameter tuning (round-safe)...")
        t = time.perf_counter()
        best_params = hyperparam_search(expert_transitions, expert_data)
        runtime["hyperparam_search_sec"] = time.perf_counter() - t
        if best_params:
            tuning_params.update(best_params)
            print(f">>> Using best params: {best_params}")
        else:
            print(">>> Search produced no improvement; keeping defaults.")

    # Vec env & learner
    venv = make_vec_env(expert_data)
    learner = build_ppo(venv, tuning_params)

    # gen_batch_size defines round size
    gen_batch_size = learner.n_steps * venv.num_envs
    eval_stride_rounds = max(1, EVAL_INTERVAL // gen_batch_size)  # cadence in rounds

    # Optional PPO pretrain on imitation reward
    if PRETRAIN_PPO and PRETRAIN_TIMESTEPS > 0:
        print(f">>> Pretraining PPO for {PRETRAIN_TIMESTEPS} steps on imitation reward...")
        t = time.perf_counter()
        # ensure pretrain is a multiple of rounds for stability (not required but neat)
        pre_steps = PRETRAIN_TIMESTEPS - (PRETRAIN_TIMESTEPS % gen_batch_size)
        pre_steps = max(pre_steps, gen_batch_size)
        n_pre_rounds = pre_steps // gen_batch_size
        for _ in range(n_pre_rounds):
            learner.learn(total_timesteps=gen_batch_size)
        runtime["pretrain_sec"] = time.perf_counter() - t

    # Reward net & AIRL
    reward_net = make_reward_net(venv)
    airl_trainer = AIRL(
        demonstrations=expert_transitions,
        demo_batch_size=256,
        gen_replay_buffer_capacity=128,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # Evaluate before training
    venv.seed(SEED)
    rewards_before, _ = evaluate_policy(learner, venv, 50, return_episode_rewards=True)

    # Training (full rounds only)
    print(">>> Training AIRL(full rounds)...")
    target_steps = TOTAL_TIMESTEPS - (TOTAL_TIMESTEPS % gen_batch_size)
    target_steps = max(target_steps, gen_batch_size)
    n_rounds = target_steps // gen_batch_size

    steps_done = 0
    metrics = {"step": [], "mean_return": [], "disc/accuracy": [], "disc/loss": [], "disc/entropy": []}

    t_train0 = time.perf_counter()
    best_reward = -np.inf
    no_improve = 0

    for r in range(n_rounds):
        airl_trainer.train(gen_batch_size)
        steps_done += gen_batch_size

        # logging cadence
        if (r + 1) % eval_stride_rounds == 0 or r == n_rounds - 1:
            logs = dict(airl_trainer.logger.name_to_value or {})
            eval_rewards, _ = evaluate_policy(learner, venv, n_eval_episodes=8, return_episode_rewards=True)
            mean_r = float(np.mean(eval_rewards))

            metrics["step"].append(steps_done)
            metrics["mean_return"].append(mean_r)
            for k_src, k_dst in [("discriminator/accuracy","disc/accuracy"),
                                 ("discriminator/loss","disc/loss"),
                                 ("discriminator/entropy","disc/entropy")]:
                v = logs.get(k_src, None)
                metrics[k_dst].append(float(v) if v is not None else np.nan)

            print(f"[{steps_done}/{target_steps}] mean_return={mean_r:.3f}, "
                  f"disc_acc={metrics['disc/accuracy'][-1]}")

            # Early stopping
            if mean_r > best_reward + 1e-6:
                best_reward = mean_r
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= EARLY_PATIENCE:
                print(f"Early stop: no improvement for {EARLY_PATIENCE} evals.")
                break

            disc_acc = metrics['disc/accuracy'][-1]
            if not np.isnan(disc_acc) and (0.5 - DISC_ACC_TOL) <= disc_acc <= (0.5 + DISC_ACC_TOL):
                print(f"Early stop: discriminator accuracy near 0.5 ±{DISC_ACC_TOL}.")
                break

    runtime["airl_train_sec"] = time.perf_counter() - t_train0

    # Save training curves
    steps_x = metrics["step"]
    if steps_x:
        plot_series(steps_x, {"mean_return": metrics["mean_return"]},
                    "Policy return vs. steps", PLOTS_DIR / "return_curve.png", ylabel="mean episodic return")
        plot_series(steps_x, {
            "disc_acc": metrics["disc/accuracy"],
            "disc_loss": metrics["disc/loss"],
            "disc_entropy": metrics["disc/entropy"]
        }, "Discriminator metrics vs. steps", PLOTS_DIR / "discriminator_curves.png")
        pd.DataFrame(metrics).to_csv(REPORTS_DIR / "training_metrics.csv", index=False)

    # Save model
    model_path = MODELS_DIR / "ppo_airl.zip"
    learner.save(str(model_path))

    # Evaluate after training
    t_eval = time.perf_counter()
    rewards_after, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    runtime["post_train_eval_sec"] = time.perf_counter() - t_eval

    # Significance
    significant = is_significant_reward_improvement(rewards_before, rewards_after, 0.001)

    # ===== Reward Signal Reporting =====
    print(">>> Reward signal reporting (expert vs generator)...")
    exp_obs = expert_transitions.obs[:5000]
    exp_acts = expert_transitions.acts[:5000]
    # gen_traj = il_rollout.generate_trajectories(
    #     policy=learner, venv=venv, sample_until=il_rollout.make_min_episodes(50)
    #      )
    gen_traj = il_rollout.generate_trajectories(
        policy=learner,
        venv=venv,
        sample_until=il_rollout.make_min_episodes(50),
        rng=np.random.default_rng(SEED),
    )

    gen_trans = il_rollout.flatten_trajectories(gen_traj)
    gen_obs = gen_trans.obs[:len(exp_obs)]
    gen_acts = gen_trans.acts[:len(exp_obs)]
    # # learned reward predictions
    exp_next = np.zeros_like(exp_obs)
    gen_next = np.zeros_like(gen_obs)
    exp_done = np.zeros(len(exp_obs), dtype=bool)
    gen_done = np.zeros(len(gen_obs), dtype=bool)

    r_exp = reward_net.predict(exp_obs, exp_acts, exp_next, exp_done).reshape(-1)
    r_gen = reward_net.predict(gen_obs, gen_acts, gen_next, gen_done).reshape(-1)

    reward_stats = {
        "expert": {"mean": float(np.nanmean(r_exp)), "std": float(np.nanstd(r_exp))},
        "generator": {"mean": float(np.nanmean(r_gen)), "std": float(np.nanstd(r_gen))},
    }
    hist_plot({"expert": r_exp, "generator": r_gen},
              "Learned reward distribution", PLOTS_DIR / "reward_histograms.png")
    save_json(reward_stats, REPORTS_DIR / "reward_stats.json")

    # ===== Test-set inference =====
    print(">>> Inference on test set and classification metrics...")
    test_data = load_expert_data(TEST_PATH)
    all_pred, all_true = [], []
    correct, total = 0, 0

    t_infer0 = time.perf_counter()
    for expert_id, trajectory in test_data.items():
        for (state, true_action, next_state, reward, done) in trajectory:
            pred_action, _ = learner.predict(state, deterministic=True)
            all_pred.append(int(pred_action))
            all_true.append(int(true_action[0]))
            correct += int(pred_action == int(true_action[0]))
            total += 1
    runtime["test_inference_sec"] = time.perf_counter() - t_infer0

    accuracy = correct / max(total, 1)
    cm = confusion_matrix(all_true, all_pred)
    clf_report = classification_report(all_true, all_pred, digits=4)

    # Confusion matrix plot
    plt.figure(figsize=(4.5,4))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0,1], ['0','1'])
    plt.yticks([0,1], ['0','1'])
    for (i,j), val in np.ndenumerate(cm):
        plt.text(j, i, val, ha='center', va='center')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Persist predictions alongside test CSV (in outputs folder)
    try:
        df_test = pd.read_csv(TEST_PATH)
        df_test['AIRL_pred'] = all_pred
        df_test.to_csv(OUT_DIR / "Prediction_with_AIRL.csv", index=False)
    except Exception as e:
        print(f"Warning: could not write prediction-augmented CSV: {e}")

    # ===== Reports =====
    summary = {
        "seed": SEED,
        "gen_batch_size": gen_batch_size,
        "steps_trained": int(steps_done),
        "pretrained": PRETRAIN_PPO,
        "pretrain_steps_effective": int(PRETRAIN_TIMESTEPS - (PRETRAIN_TIMESTEPS % gen_batch_size)) if PRETRAIN_PPO else 0,
        "training_mean_return_before": float(np.mean(rewards_before)),
        "training_std_return_before": float(np.std(rewards_before)),
        "training_mean_return_after": float(np.mean(rewards_after)),
        "training_std_return_after": float(np.std(rewards_after)),
        "significant_improvement_p001": bool(significant),
        "classification_accuracy": float(accuracy),
        "runtime_sec": runtime,
    }
    save_json(summary, REPORTS_DIR / "summary.json")
    with open(REPORTS_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("==== AIRL Training Summary ====\n")
        f.write(json.dumps(summary, indent=2, default=lambda o: int(o) if hasattr(o, "item") else str(o)))
        f.write("\n\n==== Classification report ====\n")
        f.write(clf_report)

    # Save returns arrays
    np.savez(OUT_DIR / "returns_before_after.npz",
             rewards_before=np.array(rewards_before),
             rewards_after=np.array(rewards_after))

    print("\n=== DONE ===")
    print(f"Artifacts saved under: {OUT_DIR.resolve()}")
    print(f"Mean return before: {np.mean(rewards_before):.3f} | after: {np.mean(rewards_after):.3f} "
          f"| significant @p=0.001? {'YES' if significant else 'NO'}")
    print(f"Test accuracy: {accuracy*100:.2f}%  (confusion matrix/plots in {PLOTS_DIR})")

# ===========
# CLI
# ===========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AIRL training pipeline (round-safe CLI)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--expert_train_path", type=str, default=EXPERT_TRAIN_PATH)
    parser.add_argument("--test_path", type=str, default=TEST_PATH)

    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--early_patience", type=int, default=EARLY_PATIENCE)
    parser.add_argument("--disc_acc_tol", type=float, default=DISC_ACC_TOL)

    parser.add_argument("--pretrain_ppo", action="store_true")
    parser.add_argument("--no-pretrain_ppo", dest="pretrain_ppo", action="store_false")
    parser.set_defaults(pretrain_ppo=PRETRAIN_PPO)
    parser.add_argument("--pretrain_timesteps", type=int, default=PRETRAIN_TIMESTEPS)

    parser.add_argument("--run_hparam_search", action="store_true")
    parser.add_argument("--no-run_hparam_search", dest="run_hparam_search", action="store_false")
    parser.set_defaults(run_hparam_search=RUN_HYPERPARAM_SEARCH)
    parser.add_argument("--n_trials_random_search", type=int, default=N_TRIALS_RANDOM_SEARCH)
    parser.add_argument("--search_budget_steps", type=int, default=SEARCH_BUDGET_STEPS)

    args = parser.parse_args()

    # wire CLI -> globals
    SEED = args.seed
    EXPERT_TRAIN_PATH = args.expert_train_path
    TEST_PATH = args.test_path
    TOTAL_TIMESTEPS = args.total_timesteps
    EVAL_INTERVAL = args.eval_interval
    EARLY_PATIENCE = args.early_patience
    DISC_ACC_TOL = args.disc_acc_tol
    PRETRAIN_PPO = args.pretrain_ppo
    PRETRAIN_TIMESTEPS = args.pretrain_timesteps
    RUN_HYPERPARAM_SEARCH = args.run_hparam_search
    N_TRIALS_RANDOM_SEARCH = args.n_trials_random_search
    SEARCH_BUDGET_STEPS = args.search_budget_steps

    main()
