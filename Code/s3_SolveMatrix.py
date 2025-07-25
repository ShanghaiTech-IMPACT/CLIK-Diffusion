import os
import torch
import json
import trimesh
import model.core_util as util


def solve_and_trans_mesh(target_landmarks, initial_landmarks, tooth_meshes_dict, scale=46): 
    """
    Args:
        target_landmarks: (torch.tensor): (norm-scale) [B=1, 3, 256], the landmark coordinates predicted by diffusion model
        initial_landmarks: (torch.tensor): (norm-scale) [B=1, 5, 256], the landmark coordinates (with IDs), detected by landmark detection network, serving as input
        tooth_meshes_dict: (ori-scale)
    Return:
        trans_mesh_dict
        transformation_dict
    """
    pred = target_landmarks[0]          # torch(1, 3, 256) -> (3, 256)
    init = initial_landmarks[0, :3]       # torch(1, 5, 256) -> (3, 256)

    trans_mesh_dict = {}
    transformation_dict = {}  # 4*4 transformation matrix including rotation component and translation component

    for i, name in enumerate([*range(2, 16), *range(18, 32)]):
        if name in tooth_meshes_dict.keys():
            ori_mesh = tooth_meshes_dict[name]
            ori_vert = torch.from_numpy(ori_mesh.vertices).double().cuda()   # vertices: (num, 3)  ori-scale
            ori_face = ori_mesh.faces

            start_slice = util.landmark_slices[i]
            end_slice = util.landmark_slices[i+1]
            pred_per_tooth = torch.cat((pred[:, i: i+1], pred[:, start_slice: end_slice]), dim=1)     # for each tooth, find its predicted centroid and landmarks, (3, 1+N_ldm)
            init_per_tooth = torch.cat((init[:, i: i+1], init[:, start_slice: end_slice]), dim=1)     # for each tooth, find its initial centroid and landmarks, (3, 1+N_ldm)

            # solve the rigid matrix and obtain the predicted mesh
            R, T = util.solve_rigid_matrix(X=init_per_tooth.transpose(1, 0).double(), Y=pred_per_tooth.transpose(1, 0).double())  # !!! MUST use double in GPU !!
            R, T = R.double(), T.double() * scale
            trans_vert = (R @ ori_vert.transpose(1,0)).transpose(1,0) + T   # torch(num, 3)  ori-scale
            trans_mesh = trimesh.Trimesh(vertices=trans_vert.detach().cpu().numpy(), faces=ori_face)

            trans_mesh_dict[name] = trans_mesh

            transform = torch.eye(4, dtype=torch.double)
            transform[:3, :3] = R
            transform[:3, 3] = T.flatten()
            transformation_dict[name] = transform   # ori-scale
            
    return trans_mesh_dict, transformation_dict


def save_mesh(mesh_dict, patient_name, save_dir):
    mesh_save_dir = os.path.join(save_dir, patient_name, "results")
    os.makedirs(mesh_save_dir, exist_ok=True)

    for tooth_id in mesh_dict.keys():
        tooth_mesh = mesh_dict[tooth_id]
        tooth_mesh.export(os.path.join(mesh_save_dir, f"{str(tooth_id)}.ply"))


def save_transformation(transformation_dict, patient_name, save_dir):
    mesh_save_dir = os.path.join(save_dir, patient_name, "results")
    os.makedirs(mesh_save_dir, exist_ok=True)

    json_dict = {}
    for tooth_id, trans in transformation_dict.items():
        json_dict[f"Tooth-{str(tooth_id)}"] = trans.tolist() if isinstance(trans, torch.Tensor) else trans

    with open(os.path.join(mesh_save_dir, "transformation.json"), 'w') as f:
        json.dump(json_dict, f, indent=4)