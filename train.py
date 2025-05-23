from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from Pcga_env import PCGAEnv  # Import your custom env here

# Create and wrap the environment
def make_env():
    return PCGAEnv(render_mode=None)

env = make_env()

policy_kwargs = dict(
    net_arch=[64, 64],  # Adjusted architecture for better performance
    activation_fn='sigmoid'  # Using sigmoid activation
)

# Define improved PPO agent with better hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,  # Slightly lower learning rate
    n_steps=1024,
    batch_size=256,
    ent_coef=0.03,  # Lower entropy for more exploitation
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=10,
    clip_range=0.2,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_pcga_tensorboard/"
)
# model = PPO.load("ppo_pcga_1M_steps32x32", env=env)
# model = PPO.load("ppo_pcga_1M_steps32x32_with_position_and_angle_obs", env=env)

# Train with evaluation callback
model.learn(
    total_timesteps=1_000_000
)

model.save("ppo_pcga_1M_steps64x64_with_position_and_angle_obs")
