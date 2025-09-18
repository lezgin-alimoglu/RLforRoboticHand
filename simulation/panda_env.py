# panda_env.py
import os, math, re
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

# It returns an integer based on name and objtype
def _mj_id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return -1

# Transformation matrix
def _quat_from_mat3(R):
    # rotmat -> quat (w,x,y,z)
    qw = math.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    if qw < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qx = (R[2,1] - R[1,2])/(4*qw)
    qy = (R[0,2] - R[2,0])/(4*qw)
    qz = (R[1,0] - R[0,1])/(4*qw)
    return np.array([qw, qx, qy, qz], dtype=np.float64)


class PandaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self,
                 model_path="panda_scene_with_mug.xml",
                 render=False,
                 episode_len=400,
                 ctrl_scale=0.04,          # Δq ölçeği (rad)
                 grip_speed=0.008,         # parmak hız/adımı (m)
                 lift_height=0.02,
                 place_tolerance_pos=0.03,
                 place_tolerance_yaw=0.35,
                 ee_site_hint=None,        # "panda_hand_tcp" biliyorsan ver
                 mug_body="mug",
                 target_name="mocap_target",
                 seed=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.rng   = np.random.default_rng(seed)

        self.render_enabled = render
        self.episode_len = episode_len
        self.ctrl_scale = ctrl_scale
        self.grip_speed = grip_speed
        self.lift_height = lift_height
        self.place_tol_pos = place_tolerance_pos
        self.place_tol_yaw = place_tolerance_yaw

        # --- Aktüatör -> joint eşlemesi (joint-id tabanlı, isimden bağımsız) ---
        act_by_jid = {}          # joint_id -> actuator_index
        act_joint_name = {}      # actuator_index -> joint_name (debug için)

        for ai in range(self.model.nu):
            # actuator_trnid: (nu, 2) => [joint_id, -1] (joint transmission ise)
            jid = int(self.model.actuator_trnid[ai, 0])
            if jid >= 0:
                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                act_by_jid[jid] = ai
                act_joint_name[ai] = jname

        # İsimler None olabilir: her zaman string olsun
        all_joint_names = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            all_joint_names.append(name)

        finger_regex = re.compile(r"finger", re.IGNORECASE)
        # Sadece boş olmayan isimlerde ara
        finger_joint_names = [jn for jn in all_joint_names if jn and finger_regex.search(jn)]

        
        # FIXED: act_by_jid kullanarak arm joint'leri bul
        arm_joint_names = []
        for jn in all_joint_names:
            jid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0 and jid in act_by_jid and jn not in finger_joint_names:
                arm_joint_names.append(jn)

        # Eğer finger bulunamadıysa yedek strateji:
        # - 8 aktüatör varsa son 1-2 tanesi genellikle gripper; joint tipine de bakabiliriz:
        if len(finger_joint_names) < 2:
            # SLIDE + aktüatörü olan joint'leri doğrudan joint-id üzerinden seç
            finger_jids_slide = [j for j in range(self.model.njnt)
                                if (self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE) and (j in act_by_jid)]
            finger_jids_slide = finger_jids_slide[:2]
            # İsim listesi
            finger_joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in finger_jids_slide]

            # arm joints'i yeniden hesapla (isimler boş olabilir; önemli olan birazdan JID alacağız)
            arm_joint_names = []
            for j in range(self.model.njnt):
                # aktüatörü olan ve finger setinde olmayanlar
                if (j in act_by_jid) and (j not in finger_jids_slide):
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                    arm_joint_names.append(name)

            
            # arm joints'i yeniden hesapla
            arm_joint_names = []
            for jn in all_joint_names:
                jid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0 and jid in act_by_jid and jn not in finger_joint_names:
                    arm_joint_names.append(jn)

        # Son güvenli yedek: aktüatör sayısı 8 ise ilk 7'yi kol, kalan(lar)ı finger al
        if (len(finger_joint_names) < 2 or len(arm_joint_names) < 7) and self.model.nu >= 8:
            ordered_acts = list(range(self.model.nu))
            maybe_arm_acts = ordered_acts[:7]
            maybe_finger_acts = ordered_acts[7:]
            arm_joint_names = [act_joint_name[a] for a in maybe_arm_acts if a in act_joint_name]
            finger_joint_names = [act_joint_name[a] for a in maybe_finger_acts if a in act_joint_name]

        # ID ve adresler
        self.arm_joints = arm_joint_names[:7]
        self.fingers = finger_joint_names[:2]
        self.arm_jids = [_mj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.arm_joints]
        self.finger_jids = [_mj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.fingers]
        assert all(j >= 0 for j in self.arm_jids), f"Kol eklemleri bulunamadı: {self.arm_joints}"
        assert all(j >= 0 for j in self.finger_jids), f"Parmak eklemleri bulunamadı: {self.fingers}"

        self.arm_qpos_addrs = [self.model.jnt_qposadr[j] for j in self.arm_jids]
        self.finger_qpos_addrs = [self.model.jnt_qposadr[j] for j in self.finger_jids]

        
        # Aktüatör indeksleri
        self.arm_act_indices = [act_by_jid[jid] for jid in self.arm_jids]

        # Gripper: bazı modellerde tek parmakta aktüatör var (diğeri equality ile takip eder)
        self.finger_act_indices = []
        for jid in self.finger_jids:
            ai = act_by_jid.get(jid, None)
            if ai is not None:
                self.finger_act_indices.append(ai)

        # en az bir gripper actuator bulunmalı
        assert len(self.finger_act_indices) >= 1, "Gripper actuator bulunamadı."
        # birden fazlaysa, ilkini kullan (çoğu sahnede 1 tane olur)
        self.finger_act_indices = self.finger_act_indices[:1]


        # EE site otomatik seçimi (ipucu verilmişse onu kullan)
        self.ee_site = None
        if ee_site_hint and _mj_id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_hint) >= 0:
            self.ee_site = ee_site_hint
        else:
            # adı tcp/ee/tool geçen site'lerden ilkini tercih et
            candidates = []
            for i in range(self.model.nsite):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
                if re.search(r"(tcp|ee|tool)", name, re.IGNORECASE):
                    candidates.append(name)
            self.ee_site = candidates[0] if candidates else mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, 0)

        # nesne/target isimleri
        self.mug_body = mug_body
        self.target_name = target_name

        # Action/Observation
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        obs_dim = (7+2) + (7+2) + (3+4) + (3+4) + 3
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.t = 0
        self.viewer = None  # Initialize viewer as None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Bilgi amaçlı bir satır:
        print("[ENV] arm_joints:", self.arm_joints)
        print("[ENV] finger_joints:", self.fingers)
        print("[ENV] ee_site:", self.ee_site)
        print("[ENV] arm_act_indices:", self.arm_act_indices)
        print("[ENV] finger_act_indices:", self.finger_act_indices)



    # ------------ yardımcılar ------------

    def _freejoint_addr(self, body_name):
        """Verilen body'ye bağlı FREE joint'in qpos ve qvel başlangıç adreslerini döndürür.
        qpos: 7 eleman (x,y,z,qw,qx,qy,qz), qvel: 6 eleman (vx,vy,vz, wx,wy,wz).
        FREE joint yoksa (None, None) döner.
        """
        bid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return None, None

        j_start = int(self.model.body_jntadr[bid])
        j_num   = int(self.model.body_jntnum[bid])

        for k in range(j_num):
            jid = j_start + k
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = int(self.model.jnt_qposadr[jid])  # 7 uzunluk
                dadr = int(self.model.jnt_dofadr[jid])   # 6 uzunluk
                return qadr, dadr

        return None, None

    def _body_pose(self, name):
        bid = self.model.body(name).id
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def _site_pose(self, name):
        sid = self.model.site(name).id
        pos = self.data.site_xpos[sid].copy()
        R = self.data.site_xmat[sid].reshape(3, 3).copy()
        quat = _quat_from_mat3(R)
        return pos, quat

    def _ee_pose(self):
        return self._site_pose(self.ee_site)

    def _mug_pose(self):
        return self._body_pose(self.mug_body)

    def _target_pos(self):
        sid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.target_name)
        if sid >= 0:
            return self.data.site_xpos[sid].copy()
        bid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.target_name)
        if bid >= 0:
            return self.data.xpos[bid].copy()
        # yoksa orijin
        return np.zeros(3, dtype=np.float64)

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        arm_q = np.array([qpos[adr] for adr in self.arm_qpos_addrs], dtype=np.float32)
        arm_d = np.array([qvel[self.model.jnt_dofadr[jid]] for jid in self.arm_jids], dtype=np.float32)
        finger_q = np.array([qpos[adr] for adr in self.finger_qpos_addrs], dtype=np.float32)
        finger_d = np.array([qvel[self.model.jnt_dofadr[jid]] for jid in self.finger_jids], dtype=np.float32)
        ee_p, ee_q = self._ee_pose()
        mug_p, mug_q = self._mug_pose()
        tgt_p = self._target_pos()
        obs = np.concatenate([
            arm_q, finger_q,
            arm_d, finger_d,
            ee_p, ee_q,
            mug_p, mug_q,
            tgt_p
        ]).astype(np.float32)
        return obs

    # ------------ RL API ------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        # 'home' keyframe varsa uygula
        k = _mj_id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if k >= 0:
            self.data.qpos[:self.model.nq] = self.model.key_qpos[k]
            if self.model.nu > 0:
                self.data.ctrl[:] = 0.0

        # bardağı hafif rasgeleleştir
        mug_bid = _mj_id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.mug_body)
        if mug_bid >= 0:
            base = np.array([0.45, 0.0, 0.10])
            jitter = np.array([self.rng.uniform(-0.04, 0.04),
                            self.rng.uniform(-0.04, 0.04),
                            0.0])
            qadr, dadr = self._freejoint_addr(self.mug_body)
            if qadr is not None:
                # qpos: [x y z qw qx qy qz]
                self.data.qpos[qadr:qadr+3] = base + jitter
                self.data.qpos[qadr+3:qadr+7] = np.array([1, 0, 0, 0], dtype=np.float64)
            if dadr is not None:
                # qvel: 6 boyut
                self.data.qvel[dadr:dadr+6] = 0.0


        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        dq = action[:7] * self.ctrl_scale
        gcmd = float(action[7])

        target_q = self.data.qpos.copy()

        # kol eklemleri
        for i, adr in enumerate(self.arm_qpos_addrs):
            target_q[adr] = target_q[adr] + dq[i]

        # parmaklar (simetrik açık/kapalı)
        finger_opening = 0.04
        finger_closing = 0.0
        for adr in self.finger_qpos_addrs:
            cur = target_q[adr]
            cur += self.grip_speed * (-1.0 if gcmd > 0 else +1.0)
            target_q[adr] = np.clip(cur, finger_closing, finger_opening)

        # ctrl vektörüne yaz (position actuator varsayımı)
        target_ctrl = self.data.ctrl.copy()
        
        # Arm actuators
        for i, ai in enumerate(self.arm_act_indices):
            if i < len(self.arm_qpos_addrs) and ai < len(target_ctrl):
                adr = self.arm_qpos_addrs[i]
                target_ctrl[ai] = target_q[adr]
        
        # Finger actuators        
        for i, ai in enumerate(self.finger_act_indices):
            if i < len(self.finger_qpos_addrs) and ai < len(target_ctrl):
                adr = self.finger_qpos_addrs[i]
                target_ctrl[ai] = target_q[adr]

        self.data.ctrl[:] = target_ctrl
        mujoco.mj_step(self.model, self.data)

        # --- ödül ---
        ee_p, ee_q = self._ee_pose()
        mug_p, mug_q = self._mug_pose()
        tgt_p = self._target_pos()

        dist_ee_mug = np.linalg.norm(ee_p - mug_p)
        dist_mug_tgt = np.linalg.norm(mug_p - tgt_p)

        # Gripper control - sadece 1 finger joint var
        if len(self.finger_qpos_addrs) > 0:
            grip_pos = self.data.qpos[self.finger_qpos_addrs[0]]
            grasped = (grip_pos < 0.02) and (dist_ee_mug < 0.05)
        else:
            grasped = dist_ee_mug < 0.03  # proximity-based grasping fallback
            
        lifted = (mug_p[2] > (0.10 + self.lift_height))
        placed = (dist_mug_tgt < self.place_tol_pos)

        r_reach = 1.0/(1.0 + dist_ee_mug*10.0)
        r_transport = 1.0/(1.0 + dist_mug_tgt*8.0)
        r_grasp = 0.5 if grasped else 0.0
        r_lift  = 1.0 if lifted else 0.0
        r_place = 3.0 if placed and lifted else (1.0 if placed else 0.0)

        reward = 0.5*r_reach + 0.8*r_transport + r_grasp + r_lift + r_place
        reward -= 0.001*np.sum(np.square(action))

        self.t += 1
        terminated = bool(placed and lifted)
        truncated = bool(self.t >= self.episode_len or np.any(~np.isfinite(self.data.qpos)))

        info = dict(dist_ee_mug=float(dist_ee_mug),
                    dist_mug_tgt=float(dist_mug_tgt),
                    grasped=bool(grasped),
                    lifted=bool(lifted),
                    placed=bool(placed))
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None