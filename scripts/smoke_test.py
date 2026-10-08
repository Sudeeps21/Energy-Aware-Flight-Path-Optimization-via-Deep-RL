import time
from src.envs.heavy_lift_env import HeavyLiftBatteryEnv

def test_environment():
    print("Initializing HeavyLiftBatteryEnv with PyBullet GUI...")
    # Initialize environment with GUI enabled so you can watch the drone
    env = HeavyLiftBatteryEnv(gui=True)

    obs = env.reset()
    print(f"Environment reset successful! Initial Observation shape: {obs.shape}")

    print("Running 200 random simulation steps...")
    for step in range(200):
        # Sample random actions from the action space
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 40 == 0:
            print(f"Step {step:3d} | Voltage: {info['battery_voltage']:5.2f}V | "
                  f"SoC: {info['battery_soc']*100:5.1f}% | "
                  f"Temp: {info['battery_temp']:4.1f}°C | Reward: {reward:5.2f}")

        if terminated or truncated:
            print(f"Episode ended early at step {step} due to safety cutoff!")
            obs = env.reset()

    env.close()
    print("Smoke test completed successfully!")

if __name__ == "__main__":
    test_environment()