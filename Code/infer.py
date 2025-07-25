import os
import argparse
from model.core_util import set_seed
from s1_LandmarkDetection import load_detection, detect_one_patient
from s2_LandmarkDiffusion import load_diffusion, organize_input, query_points, diffusion_one_patient
from s3_SolveMatrix import solve_and_trans_mesh, save_mesh, save_transformation



def main(args):
    set_seed(args.seed)
    patient_name = os.path.basename(args.patient_dir)

    # ================================================ Stage1 ================================================
    print("Start Stage1: Landmark Detection!")
    # Stage1: load checkpoints for the landmark detection networks
    model_incisor, model_cuspid, model_premolar, model_molar = load_detection(
        args.incisor_ckpt, args.cuspid_ckpt, args.premolar_ckpt, args.molar_ckpt
    )

    # Stage1: detect landmarks for one patient
    landmarks_dict, tooth_meshes_dict = detect_one_patient(
        patient_name,
        args.patient_dir,
        model_incisor,
        model_cuspid,
        model_premolar,
        model_molar,
        num_samples=2048,
        save_dir=(args.save_dir if args.visualize_landmarks else None)  # Whether to save and visualize the detected landmarks.
    )
    # ================================================ Stage1 ================================================


    # ================================================ Stage2 ================================================
    print("Start Stage2: Landmark Prediction via Diffusion Model!")
    # Stage2: prepare the input of the landmark-level diffusion model
    input = organize_input(landmarks_dict, tooth_meshes_dict)     # np(256, 5), consisting of 3D coordinates and IDs
    descriptor = query_points(input, tooth_meshes_dict)  # np(256, Nsample*3)

    # Stage2: load checkpoints for the landmark-level diffusion model
    network = load_diffusion(args.diffusion_ckpt)

    # Stage2: predict the post-orthodontic landmarks
    initial_landmarks, target_landmarks = diffusion_one_patient(
        input,
        descriptor,
        network
    )  # initial_landmarks: torch(1, 5, 256), target_landmarks: torch(1, 3, 256)
    # ================================================ Stage2 ================================================


    # ================================================ Stage3 ================================================
    print("Start Stage3: Rigid Transformation!")
    # Stage3: solve the rigid transformation matrix, and then apply transformation matrix into each initial tooth mesh
    trans_mesh_dict, transformation_dict = solve_and_trans_mesh(target_landmarks, initial_landmarks, tooth_meshes_dict)

    # Stage3: save transformed tooth meshes and transformation matrices
    save_mesh(trans_mesh_dict, patient_name, args.save_dir)
    save_transformation(transformation_dict, patient_name, args.save_dir)
    # ================================================ Stage3 ================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-i', '--patient_dir', type=str, help="the directory of patient", required=True)  # "Data/227"
    parser.add_argument('-o', '--save_dir', type=str, help="the directory to save results", default="Output")
    parser.add_argument('-v', '--visualize_landmarks', help="whether to save and visualize the landmarks detected in Stage1", action="store_true")

    parser.add_argument('--incisor_ckpt', type=str, help="the checkpoint of incisor landmark detection network", default="Code/checkpoint/incisor-e965.pt")
    parser.add_argument('--cuspid_ckpt', type=str, help="the checkpoint of cuspid landmark detection network", default="Code/checkpoint/cuspid-e920.pt")
    parser.add_argument('--premolar_ckpt', type=str, help="the checkpoint of premolar landmark detection network", default="Code/checkpoint/premolar-e999.pt")
    parser.add_argument('--molar_ckpt', type=str, help="the checkpoint of molar landmark detection network", default="Code/checkpoint/molar-e951.pt")
    parser.add_argument('--diffusion_ckpt', type=str, help="the checkpoint of diffusion model", default="Code/checkpoint/diffusion-e20000.pth")

    args = parser.parse_args()

    main(args)