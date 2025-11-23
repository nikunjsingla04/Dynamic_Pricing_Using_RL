"""
dynamic_pricing_project.py

Complete end-to-end example:
- Loads /mnt/data/dynamic_pricing.csv (your uploaded file)
- Uses Number_of_Riders as demand proxy (target)
- Trains a RandomForest demand model
- Builds a Gym environment that uses the demand model for counterfactual predictions
- Trains DQN (discrete pricing) and PPO (policy gradient) agents
- Evaluates and plots comparison results

Author: Generated for your project
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import random
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# -------------------------
# Config
# -------------------------
DATA_PATH = "dynamic_pricing.csv"   # <-- your uploaded CSV path
DEMAND_COL = "Number_of_Riders"               # demand proxy
BASE_PRICE_COL = "Historical_Cost_of_Ride"    # baseline price feature
MODEL_DIR = "./saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Discrete multipliers (Option A - discrete)
PRICE_MULTIPLIERS = [0.8, 0.9, 1.0, 1.1, 1.2]

# RL training params (quick defaults; increase timesteps for better performance)
DQN_TIMESTEPS = 10000
PPO_TIMESTEPS = 10000

# Evaluation
EVAL_EPISODES = 100
EPISODE_LENGTH = 30  # steps per episode (each step: one ride/context sampled)

# -------------------------
# 1) Load dataset & basic checks
# -------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check expected columns presence
if DEMAND_COL not in df.columns:
    raise RuntimeError(f"Demand column '{DEMAND_COL}' not found in CSV. Edit DEMAND_COL accordingly.")
if BASE_PRICE_COL not in df.columns:
    raise RuntimeError(f"Price column '{BASE_PRICE_COL}' not found in CSV. Edit BASE_PRICE_COL accordingly.")

# Quick view
print(df[[DEMAND_COL, BASE_PRICE_COL]].describe().T)

# -------------------------
# 2) Preprocessing: choose features
# -------------------------
# We'll use baseline price + contextual features
# Exclude target and any id-like columns
drop_cols = [DEMAND_COL]  # target
# If data contains columns that are not helpful (IDs), remove them manually here if needed

feature_cols = [c for c in df.columns if c not in drop_cols]
# We'll keep the price column too (Historical_Cost_of_Ride) - the RL agent will alter it by multiplier at inference time.

# Identify numeric vs categorical
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)

# For reproducibility, fill missing numeric by median and categorical by "missing"
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[cat_cols] = df[cat_cols].fillna("missing")

# -------------------------
# 3) Build demand model (Random Forest Regressor)
# -------------------------
X = df[feature_cols].copy()
y = df[DEMAND_COL].astype(float).copy()

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

model_pipeline = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training demand model...")
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("Demand model performance (test):")
print("  RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("  R2  :", r2_score(y_test, y_pred))

# Save model & preprocessor
joblib.dump(model_pipeline, os.path.join(MODEL_DIR, "demand_model.joblib"))
print("Saved demand model to", os.path.join(MODEL_DIR, "demand_model.joblib"))

# -------------------------
# 4) Build the Gym environment
# -------------------------
class DataDrivenPricingEnv(gym.Env):
    """
    Environment where:
    - each step samples a random context (a row from the dataset)
    - action = index into PRICE_MULTIPLIERS
    - chosen price = multiplier * historical_price
    - demand prediction is provided by the trained demand_model
    - reward = chosen_price * predicted_demand  (revenue)
    - observation = processed context features with historical price replaced by baseline price (normalized)
    - episode length is EPISODE_LENGTH
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, dataset: pd.DataFrame, model_pipeline, multipliers=PRICE_MULTIPLIERS, episode_length=EPISODE_LENGTH):
        super().__init__()
        self.raw_df = dataset.reset_index(drop=True)
        self.model = model_pipeline
        self.multipliers = multipliers
        self.episode_length = episode_length
        self.current_step = 0
        self.indices = list(self.raw_df.index)

        # Build observation space shape: we'll use preprocessor to transform an example row to get observation dimension
        # Create a "canonical" preprocessed vector for shape
        example = self.raw_df.iloc[[0]].copy()
        # observation uses same features as model input BUT we will replace Historical_Cost_of_Ride with baseline price in raw features.
        X_example = example[feature_cols]
        encoded = self.model.named_steps['pre'].transform(X_example)  # sparse or dense
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        obs_dim = encoded.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.multipliers))
        self._last_obs = None

    def _sample_context(self):
        # sample a random row index
        idx = random.choice(self.indices)
        row = self.raw_df.loc[idx].copy()
        return row

    def _encode_row(self, row):
        # row is a Series; create DataFrame for preprocessor
        X_row = pd.DataFrame([row[feature_cols]])
        enc = self.model.named_steps['pre'].transform(X_row)
        if hasattr(enc, "toarray"):
            enc = enc.toarray()
        return enc.flatten()

    def reset(self, seed=None, options=None):
        """
        Gymnasium-compatible reset returning (obs, info).
        Accepts seed and options for SB3/DummyVecEnv compatibility.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        self.context_row = self._sample_context()
        obs = self._encode_row(self.context_row).astype(np.float32)

        # Return a 2-tuple: (observation, info dict)
        return obs, {}


    def step(self, action):
        multiplier = self.multipliers[action]

        baseline_price = float(self.context_row[BASE_PRICE_COL])
        chosen_price = baseline_price * multiplier

        # Copy row & set new price
        row_with_price = self.context_row.copy()
        row_with_price[BASE_PRICE_COL] = chosen_price

        # Predict demand
        X_row = pd.DataFrame([row_with_price[feature_cols]])
        pred = self.model.predict(X_row)[0]
        predicted_demand = max(0.0, pred)

        # Reward = revenue
        revenue = chosen_price * predicted_demand

        # Step counter
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False  # we are not using truncation logic

        # Sample next context
        self.context_row = self._sample_context()
        obs = self._encode_row(self.context_row).astype(np.float32)

        info = {
            "chosen_price": chosen_price,
            "predicted_demand": predicted_demand,
            "revenue": revenue,
            "multiplier": multiplier
        }

        return obs, float(revenue), terminated, truncated, info


    def render(self, mode="human"):
        print("Last obs shape:", None if self._last_obs is None else self._last_obs.shape)

# Create env
env = DataDrivenPricingEnv(df, model_pipeline, multipliers=PRICE_MULTIPLIERS, episode_length=EPISODE_LENGTH)

# Sanity check env
obs, info = env.reset()
print("Observation shape:", obs.shape)

sample_action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(sample_action)
print("Sample step -> reward:", reward, "info:", info)


# -------------------------
# 5) Train DQN agent (discrete)
# -------------------------
# Wrap env for SB3 compatibility (it expects gym.Env)
train_env = DataDrivenPricingEnv(df, model_pipeline, multipliers=PRICE_MULTIPLIERS, episode_length=EPISODE_LENGTH)

print("\nTraining DQN agent...")
dqn_model = DQN("MlpPolicy", train_env, verbose=1, learning_rate=1e-3, buffer_size=5000)
dqn_model.learn(total_timesteps=DQN_TIMESTEPS)
dqn_model.save(os.path.join(MODEL_DIR, "dqn_pricing"))

# -------------------------
# 6) Train PPO agent (treat action as discrete with Categorical observation)
#    We'll use the same discrete action space but PPO (policy-gradient) can also work.
# -------------------------
train_env2 = DataDrivenPricingEnv(df, model_pipeline, multipliers=PRICE_MULTIPLIERS, episode_length=EPISODE_LENGTH)

print("\nTraining PPO agent...")
ppo_model = PPO("MlpPolicy", train_env2, verbose=1, learning_rate=3e-4)
ppo_model.learn(total_timesteps=PPO_TIMESTEPS)
ppo_model.save(os.path.join(MODEL_DIR, "ppo_pricing"))

# -------------------------
# 7) Evaluate both agents
# -------------------------
def evaluate_agent(model, env, episodes=100):
    revenues = []
    multipliers_chosen = []
    for ep in range(episodes):
        obs = env.reset()
        total_rev = 0.0
        for step in range(env.episode_length):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            total_rev += reward
            multipliers_chosen.append(info["multiplier"])
            if done:
                break
        revenues.append(total_rev)
    return np.array(revenues), np.array(multipliers_chosen)

# Evaluate DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return DataDrivenPricingEnv(df, model_pipeline, multipliers=PRICE_MULTIPLIERS, episode_length=EPISODE_LENGTH)

eval_env_dqn = DummyVecEnv([make_env])
eval_env_ppo = DummyVecEnv([make_env])
baseline_env = DummyVecEnv([make_env])

dqn_revs, dqn_mults = evaluate_agent(dqn_model, eval_env_dqn, episodes=EVAL_EPISODES)
print("DQN mean revenue:", dqn_revs.mean(), "std:", dqn_revs.std())

# Evaluate PPO
eval_env_ppo = DataDrivenPricingEnv(df, model_pipeline, multipliers=PRICE_MULTIPLIERS, episode_length=EPISODE_LENGTH)
ppo_revs, ppo_mults = evaluate_agent(ppo_model, eval_env_ppo, episodes=EVAL_EPISODES)
print("PPO mean revenue:", ppo_revs.mean(), "std:", ppo_revs.std())

# Baseline: historical pricing policy (use multiplier 1.0 always)
def baseline_policy(env, episodes=100):
    rets = []
    for ep in range(episodes):
        obs = env.reset()
        total_rev = 0.0
        for step in range(env.episode_length):
            # choose multiplier index corresponding to 1.0
            idx = PRICE_MULTIPLIERS.index(1.0) if 1.0 in PRICE_MULTIPLIERS else 0
            obs, reward, done, info = env.step(idx)
            total_rev += reward
            if done:
                break
        rets.append(total_rev)
    return np.array(rets)

baseline_env = DataDrivenPricingEnv(df, model_pipeline, multipliers=PRICE_MULTIPLIERS, episode_length=EPISODE_LENGTH)
baseline_revs = baseline_policy(baseline_env, episodes=EVAL_EPISODES)
print("Baseline (historical price) mean revenue:", baseline_revs.mean(), "std:", baseline_revs.std())

# -------------------------
# 8) Plots & comparison
# -------------------------
plt.figure(figsize=(8,5))
plt.boxplot([baseline_revs, dqn_revs, ppo_revs], labels=["Baseline(1.0x)", "DQN", "PPO"])
plt.ylabel("Total revenue per episode")
plt.title("Revenue comparison ({}-step episodes)".format(EPISODE_LENGTH))
plt.grid(True)
plt.savefig("revenue_comparison_boxplot.png")
plt.show()

# Plot average multiplier chosen by agents
def average_multiplier(chosen_mults):
    vals = []
    for m in PRICE_MULTIPLIERS:
        vals.append(np.mean(chosen_mults == m))
    return vals

# For plotting multipliers frequency we need the arrays per agent (we collected them in evaluate_agent)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("DQN multiplier frequency")
plt.bar([str(m) for m in PRICE_MULTIPLIERS], [np.mean(dqn_mults == m) for m in PRICE_MULTIPLIERS])
plt.ylabel("Fraction of actions")
plt.subplot(1,2,2)
plt.title("PPO multiplier frequency")
plt.bar([str(m) for m in PRICE_MULTIPLIERS], [np.mean(ppo_mults == m) for m in PRICE_MULTIPLIERS])
plt.savefig("multiplier_frequency.png")
plt.show()

# Save summary CSV
summary_df = pd.DataFrame({
    "policy": ["baseline", "dqn", "ppo"],
    "mean_revenue": [baseline_revs.mean(), dqn_revs.mean(), ppo_revs.mean()],
    "std_revenue": [baseline_revs.std(), dqn_revs.std(), ppo_revs.std()]
})
summary_df.to_csv("evaluation_summary.csv", index=False)
print("Saved evaluation_summary.csv")

print("\nAll done. Models saved in", MODEL_DIR)
print("Saved plots: revenue_comparison_boxplot.png, multiplier_frequency.png")
