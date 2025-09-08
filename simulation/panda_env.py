# panda_env.py
import os
import numpy as np
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("pip install gymnasium")

# Panda dosya yolu (scene.xml)
# Önce ENV değişkenine bak, yoksa ../panda_model/scene.xml dene
DEFAULT_SCENE = os.environ.get(
    "PANDA_SCENE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models/panda_model", "scene.xml"))
)

class PandaReachEnv(gym.Env):
    """
    Minimal Panda ortamı:
      - Action: 7 eklem için [-0.05, 0.05] rad pozisyon artışı
      - Observation: [q(7), dq(7)]
      - Reward: uç-efektörü (varsayılan site) hedefe yaklaştırma (opsiyonel)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, scene_path: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        scene = scene_path or DEFAULT_SCENE
        if not os.path.exists(scene):
            raise FileNotFoundError(f"Scene not found: {scene}")

        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)

        # Panda kolu 7 DOF (ilk 7 qpos/qvel)
        self.nq = 7
        self.nu = 7
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(self.nu,), dtype=np.float32)
        obs_high = np.inf * np.ones(self.nq + self.nq, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Control timing
        self.ctrl_dt = 0.01  # 100 Hz
        self._ctrl_skip = max(1, int(self.ctrl_dt / self.model.opt.timestep))
        self.max_steps = 300
        self._steps = 0

        # Viewer / renderer lazımsa oluştur
        self._viewer = None
        self._renderer = None

    def _get_obs(self):
        q = self.data.qpos[:self.nq].copy()
        dq = self.data.qvel[:self.nq].copy()
        return np.concatenate([q, dq]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # nötr çevresinde küçük gürültü
        self.data.qpos[:self.nq] += 0.01 * self.np_random.standard_normal(self.nq)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0

        if self.render_mode == "human":
            self._ensure_viewer()
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # basit position "servo": qpos = qpos + action
        q_target = self.data.qpos[:self.nq] + action

        for _ in range(self._ctrl_skip):
            self.data.qpos[:self.nq] = q_target
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

        self._steps += 1

        # çok basit bir ödül: eklem hızlarını küçük tut
        reward = -0.01 * np.linalg.norm(self.data.qvel[:self.nq])

        terminated = False
        truncated = self._steps >= self.max_steps
        info = {}
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 640, 480)
            self._renderer.update_scene(self.data)
            rgb, _ = self._renderer.read_pixels()
            return rgb
        elif self.render_mode == "human":
            self._ensure_viewer()
            self._render_frame()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer = None

    def _ensure_viewer(self):
        if self._viewer is None:
            from mujoco import viewer
            self._viewer = viewer.launch_passive(self.model, self.data)

    def _render_frame(self):
        if self._viewer is not None:
            self._viewer.sync()
