import numpy as np

def play_with_pos_vel(mocap_filepath):
    from mujoco_py import load_model_from_xml, MjSim, MjViewer

    xmlpath = '/home/kenji/Fyp/DeepMimic_mujoco/diffusion/mujoco/humanoid_deepmimic/envs/asset/dp_env_v2.xml'
    with open(xmlpath) as fin:
        MODEL_XML = fin.read()

    model = load_model_from_xml(MODEL_XML)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    # Load the mocap
    mocap = np.load(mocap_filepath)
    print(mocap.shape)  
    from time import sleep

    phase_offset = np.array([0.0, 0.0, 0.0])

    # import cv2
    # from VideoSaver import VideoSaver
    # width = 640
    # height = 480

    # vid_save = VideoSaver(width=width, height=height)
    while True:
    # for i in range(5):
        for frame in mocap: 
            data_config, data_vel = frame[:35], frame[35:]
            print(len(data_config), len(data_vel))
            tmp_val = data_config
            sim_state = sim.get_state()
            sim_state.qpos[:] = tmp_val[:]
            # sim_state.qpos[:3] +=  phase_offset[:]
            sim.set_state(sim_state)
            sim.forward()
            viewer.render()
            # vid_save.addFrame(viewer.read_pixels(width, height, depth=False))

        sim_state = sim.get_state()
        phase_offset = sim_state.qpos[:3]
        phase_offset[2] = 0

    # vid_save.close()

def play_with_frame_data(mocap_filepath):
    from mujoco_py import load_model_from_xml, MjSim, MjViewer
    from diffusion.utils.mocap_v2 import MocapDM

    xmlpath = '/home/kenji/Fyp/DeepMimic_mujoco/diffusion/mujoco/humanoid_deepmimic/envs/asset/dp_env_v2.xml'
    with open(xmlpath) as fin:
        MODEL_XML = fin.read()

    model = load_model_from_xml(MODEL_XML)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    # Load the mocap
    mocap = np.load(mocap_filepath)
    mocap_dm = MocapDM()
    mocap_dm.load_mocap_from_raw(mocap)
    from time import sleep

    phase_offset = np.array([0.0, 0.0, 0.0])

    data_config = mocap_dm.data_config
    data_vel = mocap_dm.data_vel

    # import cv2
    # from VideoSaver import VideoSaver
    # width = 640
    # height = 480

    # vid_save = VideoSaver(width=width, height=height)
    while True:
        # for i in range(5):
        for config, vel in zip(data_config, data_vel): 
            tmp_val = config
            sim_state = sim.get_state()
            sim_state.qpos[:] = tmp_val[:]
            sim_state.qpos[:3] +=  phase_offset[:]
            sim_state.qvel[:] = vel[:]
            sim.set_state(sim_state)
            sim.forward()
            viewer.render()
            # vid_save.addFrame(viewer.read_pixels(width, height, depth=False))

        sim_state = sim.get_state()
        phase_offset = sim_state.qpos[:3]
        phase_offset[2] = 0

    # vid_save.close()

if __name__ == '__main__':
    play_with_frame_data(
        "/home/kenji/Fyp/DeepMimic_mujoco/diffusion/logs/report-dissertation-3/sampled_motions/motion_2.npy"
    )
