from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Pcga_env import PCGAEnv

# Create and wrap environment
def make_env():
    env = PCGAEnv(render_mode="human")
    env.pcga.set_target_position(130.6)
    return env

env = DummyVecEnv([make_env])

# Load trained model

model = PPO.load("ppo_pcga_1M_steps32x32_random_improved", env=env)
# model = PPO.load("ppo_pcga_1M_steps64x64_with_position_and_angle_obs", env=env)

# Initialize
obs = env.reset()
step_count = 0
max_steps = 10000

while step_count < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step_count} — Action: {action}")
    obs, reward, done, info = env.step(action)
    env.render()
    step_count += 1
    if done:
        obs = env.reset()

