# train_panda_pickplace.py
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from panda_env import PandaPickPlaceEnv

def make_env():
    return PandaPickPlaceEnv(
        model_path="models/main.xml",   
        render=False,
        episode_len=2000,
        ctrl_scale=0.01,
        mug_body="mug",
        target_name="mocap_target",
        ee_site_hint="gripper"           
    )


if __name__ == "__main__":

    os.makedirs("./ckpts", exist_ok=True)
    
    env = make_env()
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200000,
        batch_size=256,
        tau=0.02,
        gamma=0.98,
        train_freq=64,
        gradient_steps=64,
        verbose=1
    )
    
    ckpt_cb = CheckpointCallback(save_freq=50000, save_path="./ckpts", name_prefix="sac_panda")
    
    print("Starting training...")
    model.learn(total_timesteps=1_000_000, callback=ckpt_cb)
    
    print("Saving final model...")
    model.save("sac_panda_pickplace")
    
    env.close()
    print("Training completed!") 