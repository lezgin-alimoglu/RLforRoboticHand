# run_panda.py
import time
import numpy as np
from panda_env import PandaReachEnv

if __name__ == "__main__":
    # viewer ile çalıştır
    env = PandaReachEnv(render_mode="human")
    obs, info = env.reset()

    # 10 episodluk rastgele policy denemesi
    for ep in range(10):
        total_r = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            # viewer’ı rahatlatmak için küçük uyku (opsiyonel)
            time.sleep(0.001)
        print(f"Episode {ep+1}: steps={steps}, return={total_r:.3f}")
        obs, info = env.reset()

    env.close()
