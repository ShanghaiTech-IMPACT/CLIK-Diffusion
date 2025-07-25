import torch
import numpy as np
import model.core_util as util
from model.diffusion_network import Network_1dUNet as Network



diffusion_cfgs = {
    "unet": {
        "in_channel": 24,
        "out_channel": 3,
        "inner_channel": 64,
        "channel_mults": [
            1,
            2,
            4,
            8
        ],
        "attn_res": [
            16
        ],
        "num_head_channels": 32,
        "res_blocks": 2,
        "dropout": 0.2,

        "in_feature_channel": 384,
        "inner_feature_channel": 128,
        "out_feature_channel": 16
    },

    "beta_schedule": {
        "schedule": "linear",
        "n_timestep": 2000,
        "linear_start": 1e-6,
        "linear_end": 0.01
    }
}   


def organize_input(landmarks_dict, tooth_meshes_dict, scale=46):
    """
    Landmarks of the whole dentition should be organized as (256, 3).
    """
    landmarks_list = []
    centroids_list = []
    for i, name in enumerate([*range(2, 16), *range(18, 32)]):
        if name in [7, 8, 9, 10, 23, 24, 25, 26]:     # incisor
            landmark_ids = ['0', '3', '4', '13', '15', '16', '34', '35']
        elif name in [6, 11, 22, 27]:                 # cuspid
            landmark_ids = ['0', '3', '4', '5', '19', '20', '34', '35']
        elif name in [4, 5, 12, 13, 20, 21, 28, 29]:  # premolar
            landmark_ids = ['0', '4', '6', '7', '21', '22', '23', '34', '35']
        elif name in [2, 3, 14, 15, 18, 19, 30, 31]:  # molar
            landmark_ids = ['0', '8', '9', '10', '11', '21', '22', '25', '30', '34', '35']  
        else:
            raise Exception("Wrong tooth!")

        for ldm_id in landmark_ids:
            if name in tooth_meshes_dict.keys():
                ldm_coord = np.array(landmarks_dict[name][ldm_id])  # np[x, y, z]
                x, y, z = ldm_coord / scale
            else:  # vacant tooth
                x, y, z = 0., 0., 0.
            
            item = int(ldm_id) / 35
            tooth = name / 32
            ldm = np.array([x, y, z, item, tooth])
            
            if ldm_id == '0':  # centroid
                centroids_list.append(ldm)
            else: # other landmarks
                landmarks_list.append(ldm)

    landmarks = np.stack(landmarks_list, axis=0)     # np(num_ldm, 5)
    centroids = np.stack(centroids_list, axis=0)   # np(28, 5)
    input = np.concatenate([centroids, landmarks], axis=0)   # np(28+num_ldm, 5)

    return input


def query_points(landmarks, tooth_meshes_dict, Nsample=128, radius=2, scale=46):
    """
    Query Nsample points in sphere with radius=2mm (need scale) for each landamrk.
    Note that for (1) centorid: farthest point sampling through the whole tooth (2mm sphere around the centorid has no vertices)
                    (2) other landmarks: query ball points like PointNet++
    Args:
        landmarks (np.ndarray): [256, 5], includes 28 centroids + 228 other landmarks
        Nsample (int): the number of points sampled around each landmark
    Return:
        query_points (np.ndarray): the queried points around each landmark among 256, [256, Nsample*3]
    """
    landmarks = landmarks[:, :3]  # np(28+228, 3), 28 centroids + 228 other landmarks
    query_points = np.zeros((256, Nsample, 3))   # (256, Nsample, 3)
    for i, name in enumerate([*range(2, 16), *range(18, 32)]):
        if name in tooth_meshes_dict.keys():   # for vacant tooth, query_points will be zero
            start_index = util.landmark_slices[i]
            end_index = util.landmark_slices[i+1]
            landmarks_per_tooth = landmarks[start_index: end_index]   # (228, 3) -> (7, 3)

            # get the vertices of the corresponding tooth
            vertices = tooth_meshes_dict[name].vertices / scale   # np(N, 3)

            # (1) for centorid: farthest point sampling through the whole tooth
            fps_idx = util.farthest_point_sample(vertices, Nsample)  # np(Nsample)
            fps_points = vertices[fps_idx]     # np(Nsample, 3)
            # (2) for other landmarks: query ball points like PointNet++
            ball_idx = util.query_ball_point(radius=radius/scale, nsample=Nsample, points=vertices, centers=landmarks_per_tooth)  # np(7, Nsample)
            ball_points = vertices[ball_idx]   # np(7, Nsample, 3)

            # merge as query points
            query_points[i] = fps_points
            query_points[start_index: end_index] = ball_points

    return query_points.reshape(256, -1)   # (256, Nsample, 3) -> (256, Nsample*3)


def load_diffusion(ckpt):
    netG = Network(diffusion_cfgs['unet'], diffusion_cfgs['beta_schedule'])
    netG.load_state_dict(torch.load(ckpt), strict=False)
    netG.to(torch.device('cuda'))
    netG.eval()
    netG.set_new_noise_schedule()
    return netG



def diffusion_one_patient(
    input,
    descriptor,
    network
):
    """
    Args:
        input (np.ndarray): [256, 5], includes 28 centroids + 228 other landmarks, (normalize-scale)
        descriptor (np.ndarray): [256, Nsample*3], (normalize-scale)
    Return:
        output: (torch.tensor): [B=1, 3, 256], the predicted landmark coordinates
    """
    # put data on CUDA
    input = torch.from_numpy(input).float().transpose(1, 0).unsqueeze(0).cuda()   # (256, 5) -> (5, 256) -> (1, 5, 256)
    descriptor = torch.from_numpy(descriptor).float().transpose(1, 0).unsqueeze(0).cuda()  # (256, Nsample*3) -> (Nsample*3, 256) -> (1, Nsample*3, 256)

    # infer
    network.eval()
    with torch.no_grad():
        output, _ = network.restoration(
            y_cond=input, 
            y_t=torch.rand_like(input[:, :3]), 
            y_0=None,  # GT is no need during inference
            sample_num=1, 
            extra_features=descriptor
        )

    return input, output


