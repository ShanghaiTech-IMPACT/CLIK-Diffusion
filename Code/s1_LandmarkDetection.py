import os 
import torch 
import numpy as np
from glob import glob
import trimesh
import json
from natsort import natsorted
from model.point_mlp import PointMLP


def pc_normalize(pc):
    c = np.mean(pc, axis=0)
    pc = pc - c
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, c, m


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = torch.from_numpy(xyz)
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids.squeeze().numpy()








def load_detection(
    incisor_ckpt=None,
    cuspid_ckpt=None,
    premolar_ckpt=None,
    molar_ckpt=None):

    model_incisor = PointMLP(num_classes=7, points=2048, embed_dim=128)
    model_incisor.load_state_dict(torch.load(incisor_ckpt)['model'])
    model_incisor.cuda()
    model_incisor.eval()

    model_cuspid = PointMLP(num_classes=7, points=2048, embed_dim=128)
    model_cuspid.load_state_dict(torch.load(cuspid_ckpt)['model'])
    model_cuspid.cuda()
    model_cuspid.eval()

    model_premolar = PointMLP(num_classes=8, points=2048, embed_dim=128)
    model_premolar.load_state_dict(torch.load(premolar_ckpt)['model'])
    model_premolar.cuda()
    model_premolar.eval()

    model_molar = PointMLP(num_classes=10, points=2048, embed_dim=128)
    model_molar.load_state_dict(torch.load(molar_ckpt)['model'])
    model_molar.cuda()
    model_molar.eval()

    return model_incisor, model_cuspid, model_premolar, model_molar


def detect_one_patient(
    patient_name,
    folder,
    model_incisor,
    model_cuspid,
    model_premolar,
    model_molar,
    num_samples=2048,
    save_dir=None,
):
    folder = os.path.join(folder, 'initial')
    tooth_mesh_files = natsorted(glob(os.path.join(folder, '*.stl')) + glob(os.path.join(folder, '*.ply')))
    tooth_ids = [int(os.path.basename(tooth_mesh_file).split('.')[0]) for tooth_mesh_file in tooth_mesh_files]
    tooth_meshes = [trimesh.load(tooth_mesh_file) for tooth_mesh_file in tooth_mesh_files]
    tooth_meshes_dict = dict(zip(tooth_ids, tooth_meshes))

    landmarks_dict = {}
    for idx, tooth_id in enumerate(tooth_ids):
        tooth = tooth_meshes_dict[tooth_id]
        faces = tooth.faces
        ori_points, normals = tooth.vertices, tooth.vertex_normals

        points, center, scale = pc_normalize(ori_points)
        sample_idxes = farthest_point_sample(points[None], num_samples)
        sample_points = points[sample_idxes]    # (num_samples, 3)
        sample_normals = normals[sample_idxes]  # (num_samples, 3)

        points_tensor = torch.from_numpy(sample_points).float().unsqueeze(0).cuda().permute(0, 2, 1)    # (1, 3, num_samples)
        normals_tensor = torch.from_numpy(sample_normals).float().unsqueeze(0).cuda().permute(0, 2, 1)  # (1, 3, num_samples)
        tid = torch.from_numpy(np.array([tooth_id])).long().cuda()

        with torch.no_grad():
            if tooth_id in [7, 8, 9, 10, 23, 24, 25, 26]:  # incisor
                landmark_ids = ['3', '4', '13', '15', '16', '34', '35']
                model = model_incisor
            elif tooth_id in [6, 11, 22, 27]:  # cuspid
                landmark_ids = ['3', '4', '5', '19', '20', '34', '35']
                model = model_cuspid
            elif tooth_id in [4, 5, 12, 13, 20, 21, 28, 29]:  # premolar
                landmark_ids = ['4', '6', '7', '21', '22', '23', '34', '35']
                model = model_premolar
            elif tooth_id in [2, 3, 14, 15, 18, 19, 30, 31]:  # molar
                landmark_ids = ['8', '9', '10', '11', '21', '22', '25', '30', '34', '35']  
                model = model_molar
            elif tooth_id in [1, 16, 17, 32]:  # 3rd molar
                continue
            else:
                raise Exception("Wrong tooth!")

            pred = model(points_tensor, normals_tensor, tid)    # torch(1, 2048, num_landmarks)
            pred_idx = torch.argmax(pred, dim=1).squeeze().cpu().numpy()  # np(num_landmarks), 0~num_samples


        # obtain the coordinates of lanmarks according to the predicted index
        ldm_dict = {}
        ldm_dict['0'] = tuple(np.mean(np.array(tooth.vertices), 0))  # landmar'0' represents the centroid of each tooth

        for ldm_i, ldm_id in enumerate(landmark_ids):
            pred_idx_i = pred_idx[ldm_i]
            pred_coord_i = sample_points[pred_idx_i]   # np(3), (x, y, z)
            pred_coord_i = pred_coord_i * scale + center  # np(3), (x, y, z), original size
            x, y, z = float(pred_coord_i[0]), float(pred_coord_i[1]), float(pred_coord_i[2])      
            ldm_dict[ldm_id] = (x, y, z)

        landmarks_dict[tooth_id] = ldm_dict


        # [Optional] save and visualize
        if save_dir is not None:
            patient_save_dir = os.path.join(save_dir, patient_name)
            os.makedirs(patient_save_dir, exist_ok=True)

            # save the landmarks into .json
            landmark_save_dir = os.path.join(patient_save_dir, 'landmarks')
            os.makedirs(landmark_save_dir, exist_ok=True)
            with open(os.path.join(landmark_save_dir, str(tooth_id)+'.json'), 'w') as f:
                fprint = json.dumps(ldm_dict, indent=4)
                f.write(fprint)
            # print('{} == EXPORTED TO == {}'.format(tooth_mesh_file, os.path.join(landmark_save_dir, str(tooth_id)+'.json')))


            #  visualize the landmarks
            from model.core_util import visualize_pred_landmark
            landmark, tooth = visualize_pred_landmark(landmark_ids, pred_idx, sample_points.copy(), scale, center, ori_points, faces)
            visualize_save_dir = os.path.join(patient_save_dir, 'landmarks_vis')
            os.makedirs(visualize_save_dir, exist_ok=True)
            landmark.export(os.path.join(visualize_save_dir, 'Landmark_{}.obj'.format(str(tooth_id))))
            tooth.export(os.path.join(visualize_save_dir, 'Tooth_{}.obj'.format(str(tooth_id))))


    return landmarks_dict, tooth_meshes_dict  # ori-scale
