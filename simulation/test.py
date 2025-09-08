import time
import numpy as np
import mujoco
from mujoco import viewer
import os

# Path to the combined scene file
COMBINED_SCENE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "scene.xml")
)

def move_joint(data, model, joint_idx, target, steps=200):
    """Move specified joint slowly to target position"""
    if joint_idx < 0 or joint_idx >= model.nq:
        print(f"Warning: Invalid joint index {joint_idx}")
        return
        
    start = data.qpos[joint_idx]
    for alpha in np.linspace(0, 1, steps):
        data.qpos[joint_idx] = (1 - alpha) * start + alpha * target
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        yield

def find_joint_id(model, joint_name):
    """Find joint ID by name"""
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    except:
        return -1

def print_model_info(model):
    """Print detailed information about the loaded model"""
    print(f"\n=== Model Information ===")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"DOFs: {model.nv}")
    print(f"Geoms: {model.ngeom}")
    
    print(f"\n=== Joint Details ===")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type_names = {
            0: "FREE",
            1: "BALL", 
            2: "SLIDE",
            3: "HINGE"
        }
        joint_type = joint_type_names.get(model.jnt_type[i], f"TYPE_{model.jnt_type[i]}")
        qpos_adr = model.jnt_qposadr[i]
        print(f"  Joint {i}: '{joint_name}' | Type: {joint_type} | QPos: {qpos_adr}")
    
    print(f"\n=== Body Details ===")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  Body {i}: '{body_name}'")

def main():
    # Check if combined scene file exists
    if not os.path.exists(COMBINED_SCENE_PATH):
        raise FileNotFoundError(f"Combined scene not found: {COMBINED_SCENE_PATH}")
    
    try:
        # Load the combined model
        model = mujoco.MjModel.from_xml_path(COMBINED_SCENE_PATH)
        print(f"Successfully loaded combined model!")
        
        # Print model information for debugging
        print_model_info(model)
            
    except Exception as e:
        print(f"Error loading combined model: {e}")
        print("Make sure:")
        print("1. The combined_scene.xml file exists")
        print("2. The included files (scene.xml, cup.xml) exist")
        print("3. File paths in the include statements are correct")
        return
    
    data = mujoco.MjData(model)
    
    # Find specific joints
    cup_joint_id = find_joint_id(model, "cup_free")
    
    # Find Panda robot joints (adjust these names based on your actual model)
    # Common Panda joint names: panda_joint1, panda_joint2, etc.
    possible_joint_names = [
        "panda_joint1", "panda0_joint1", "joint1",
        "panda_joint3", "panda0_joint3", "joint3"
    ]
    
    joint1_id = -1
    joint3_id = -1
    
    for name in possible_joint_names:
        if "joint1" in name and joint1_id == -1:
            joint1_id = find_joint_id(model, name)
        elif "joint3" in name and joint3_id == -1:
            joint3_id = find_joint_id(model, name)
    
    # Fallback to indices if names don't work
    if joint1_id == -1:
        joint1_id = 1
        print(f"Warning: Using fallback joint index 1 for joint1")
    if joint3_id == -1:
        joint3_id = 3
        print(f"Warning: Using fallback joint index 3 for joint3")
    
    print(f"\nUsing joints - Joint1 ID: {joint1_id}, Joint3 ID: {joint3_id}")
    
    if cup_joint_id != -1:
        print(f"Cup free joint found with ID: {cup_joint_id}")
    else:
        print("Warning: Cup free joint not found - cup will be static")
    
    # Start the simulation
    with viewer.launch_passive(model, data) as v:
        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                # Reset simulation data
                mujoco.mj_resetData(model, data)
                
                # Randomize cup position if free joint exists
                if cup_joint_id != -1:
                    joint_adr = model.jnt_qposadr[cup_joint_id]
                    random_y = np.random.uniform(-0.1, 0.1)
                    random_x = np.random.uniform(0.4, 0.6)
                    cup_pos = np.array([random_x, random_y, 0.08])
                    
                    print(f"Setting cup position: [{cup_pos[0]:.2f}, {cup_pos[1]:.2f}, {cup_pos[2]:.2f}]")
                    
                    # Set position (x, y, z)
                    data.qpos[joint_adr:joint_adr+3] = cup_pos
                    # Set orientation (quaternion w, x, y, z)
                    data.qpos[joint_adr+3:joint_adr+7] = [1, 0, 0, 0]
                
                mujoco.mj_forward(model, data)
                
                # Robot arm movement sequence
                print("Phase 1: Extending arm forward...")
                for step in move_joint(data, model, joint1_id, -1.0):
                    v.sync()
                    time.sleep(0.01)
                
                print("Phase 2: Moving joint 3...")    
                for step in move_joint(data, model, joint3_id, 1.5):
                    v.sync()
                    time.sleep(0.01)
                
                time.sleep(1.0)
                
                print("Phase 3: Retracting joint 3...")
                for step in move_joint(data, model, joint3_id, 0.0):
                    v.sync()
                    time.sleep(0.01)
                
                print("Phase 4: Retracting arm...")    
                for step in move_joint(data, model, joint1_id, 0.0):
                    v.sync()
                    time.sleep(0.01)
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")

if __name__ == "__main__":
    main()