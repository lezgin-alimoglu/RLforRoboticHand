# eval_panda.py
import time
import numpy as np
from stable_baselines3 import SAC
from panda_env import PandaPickPlaceEnv

MODEL_XML = "models/main.xml"     
MODEL_ZIP = "ckpts/sac_panda_1000000_steps" 
DET = True                        
MAX_STEPS = 2000                   # Same with epsido length



def main():
    try:
        import mujoco.viewer as mjv
        has_viewer = True
    except Exception:
        mjv = None
        has_viewer = False

    env = PandaPickPlaceEnv(
        model_path=MODEL_XML,
        render=False,          
        mug_body="mug",
        target_name="mocap_target",
        ee_site_hint="gripper",
        episode_len=MAX_STEPS,
    )

    model = SAC.load(MODEL_ZIP, env=env)

    viewer = None
    if has_viewer:
        try:
            viewer = mjv.launch_passive(env.model, env.data)
            print("[eval] Mujoco viewer is opened.")
        except Exception as e:
            print("[eval] Viewer is not opened, offscreen :", e)

    # Episode begin
    obs, _ = env.reset()
    done, truncated = False, False
    ep_reward = 0.0

    try:
        for t in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=DET)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)

            if viewer is not None:
                viewer.sync()

                time.sleep(0.02)
            else:
                time.sleep(0.02)

            if done or truncated:
                break

        print(f"[eval] Episode finished | steps={t+1} | return={ep_reward:.2f}")
        if info:
            print("[eval] info:", {k: info[k] for k in ["grasped", "lifted", "placed"] if k in info})

    except KeyboardInterrupt:
        print("\n[eval] User stopped.")
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        env.close()

if __name__ == "__main__":
    main()
