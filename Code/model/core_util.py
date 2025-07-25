import random
import numpy as np
import math
import os
import torch
from scipy.spatial import cKDTree
import matplotlib
import trimesh

landmark_slices = [28, 38, 48, 56, 64, 71, 78, 85, 92, 99, 106, 114, 122, 132, 142,
                   152, 162, 170, 178, 185, 192, 199, 206, 213, 220, 228, 236, 246, 256]

### about contact points
contact1_slices = [36, 46, 54, 62, 69, 76, 83, 91, 98, 105, 113, 121, 131, 150, 160, 168, 176, 183, 190, 197, 205, 212, 219, 227, 235, 245]
contact2_slices = [47, 55, 63, 70, 77, 84, 90, 97, 104, 112, 120, 130, 140, 161, 169, 177, 184, 191, 198, 204, 211, 218, 226, 234, 244, 254]


### about vector points
vertical_slices = [list(range(28, 36)),
                 list(range(38, 46)),
                 list(range(49, 54)),
                 list(range(57, 62)),
                 list(range(66, 69)),
                 list(range(73, 76)),
                 list(range(80, 83)),
                 list(range(87, 90)),
                 list(range(94, 97)),
                 list(range(101, 104)),
                 list(range(107, 112)),
                 list(range(115, 120)),
                 list(range(122, 130)),
                 list(range(132, 140)),
                 list(range(142, 150)),
                 list(range(152, 160)),
                 list(range(163, 168)),
                 list(range(171, 176)),
                 list(range(180, 183)),
                 list(range(187, 190)),
                 list(range(194, 197)),
                 list(range(201, 204)),
                 list(range(208, 211)),
                 list(range(215, 218)),
                 list(range(221, 226)),
                 list(range(229, 234)),
                 list(range(236, 244)),
                 list(range(246, 254))]


horizontal_slices = [list(range(32, 34)),
                 list(range(42, 44)),
                 list(range(51, 53)),
                 list(range(59, 61)),
                 list(range(67, 69)),
                 list(range(74, 76)),
                 list(range(81, 83)),
                 list(range(88, 90)),
                 list(range(95, 97)),
                 list(range(102, 104)),
                 list(range(109, 111)),
                 list(range(117, 119)),
                 list(range(126, 128)),
                 list(range(136, 138)),
                 list(range(146, 148)),
                 list(range(156, 158)),
                 list(range(165, 167)),
                 list(range(173, 175)),
                 list(range(181, 183)),
                 list(range(188, 190)),
                 list(range(195, 197)),
                 list(range(202, 204)),
                 list(range(209, 211)),
                 list(range(216, 218)),
                 list(range(223, 225)),
                 list(range(231, 233)),
                 list(range(240, 242)),
                 list(range(250, 252))]



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, numpy array of shape [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled point indices, numpy array of shape [npoint]
    """
    N, C = xyz.shape
    assert npoint <= N, "Number of points to sample must be less than or equal to the number of points in the point cloud"
    indices = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        indices[i] = farthest
        centroid = xyz[farthest, :].reshape(1, C)
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)
    return indices



def query_ball_point(radius, nsample, points, centers):
    """
    Find `nsample` points in the radius of `centers` from `points`.
    Input:
        radius (float): The radius of the sphere.
        nsample (int): The number of samples in the local region.
        points (np.ndarray): All points, [N, 3].
        centers (np.ndarray): Query points, [S, 3].
    Returns:
        indices (np.ndarray): Indices of grouped points, [S, nsample].
    """
    tree = cKDTree(points)
    indices = []
    for center in centers:
        idx = tree.query_ball_point(center, radius)
        if len(idx) < nsample:    # If there are not enough points, we need to sample with replacement
            idx = np.random.choice(idx, nsample, replace=True)
        elif len(idx) > nsample:  # If there are more than enough points, we randomly choose nsample points
            idx = random.sample(idx, nsample)
        indices.append(idx)
    return np.array(indices)





def my_vander(x, N=None):
    """
    Generate Vandermonde matrix for input tensor x.

    Args:
        x (torch.Tensor): Input tensor (shape: [batch_size, num_points]).
        N (int): Number of columns in the Vandermonde matrix.

    Returns:
        vander_matrix (torch.Tensor): Vandermonde matrix (shape: [batch_size, num_points, N]).
    """
    # Vandermonde matrix formula for (x, y, z)
    # V = | 1    x    x^2    ...   x^(N-1) |
    #     | 1    y    y^2    ...   y^(N-1) |
    #     | 1    z    z^2    ...   z^(N-1) |

    if N is None:
        N = x.size(-1)
    powers = torch.arange(N, dtype=x.dtype, device=x.device).view(1, 1, -1)
    vander_matrix = x.unsqueeze(-1) ** powers

    return vander_matrix


def polyfit(x, y, degree=4):
    """
    【No missing teeth】polynomial fit using least squares for each batch. [From low-order to high-order coeffs (low to high)]

    Args:
        x (torch.Tensor): Input tensor (shape: [batch_size, num_points]).
        y (torch.Tensor): Output tensor (shape: [batch_size, num_points]).
        degree (int): Degree of the polynomial.

    Returns:
        coefficients (torch.Tensor): Fitted coefficients for each batch (shape: [batch_size, degree+1]). [From low-order to high-order coeffs]
    """

    x_vander = my_vander(x, degree + 1)
    coefficients = torch.linalg.solve(x_vander.transpose(-1, -2) @ x_vander, x_vander.transpose(-1, -2) @ y.unsqueeze(-1))

    return coefficients


def polyfit_weighted(x, y, weights, degree=4):
    """
    polynomial fit using least squares for each batch. [From low-order to high-order coeffs (low to high)]

    Args:
        x (torch.Tensor): Input tensor (shape: [batch_size, num_points]).
        y (torch.Tensor): Output tensor (shape: [batch_size, num_points]).
        weights (torch.Tensor): 1 or 0, represents if the tooth is missing (shape: [batch_size, num_points]).
        degree (int): Degree of the polynomial.

    Returns:
        coefficients (torch.Tensor): Fitted coefficients for each batch (shape: [batch_size, degree+1]). [From low-order to high-order coeffs]
    """

    x_vander = my_vander(x, degree + 1)

    weighted_A = torch.diag_embed(weights) @ x_vander
    weighted_b = weights * y

    coefficients = torch.linalg.solve(weighted_A.transpose(-1, -2) @ weighted_A, weighted_A.transpose(-1, -2) @ weighted_b.unsqueeze(-1))

    return coefficients.squeeze(-1)






def solve_rigid_matrix(X, Y):
    """
    Solve Y = RX + t

    Reference: https://zhuanlan.zhihu.com/p/111322916
    Input:
        X: Nx3 torch tensor of N points
        Y: Nx3 torch tensor of N points
    Returns:
        R: 3x3 torch tensor describing camera orientation
        t: [3] torch tensor describing camera translation

    """
    device = X.device

    # equation (2)
    cx, cy = X.sum(dim=0) / Y.shape[0], Y.mean(dim=0)

    # equation (6)
    x, y = X - cx, Y - cy

    # equation (13)
    w = torch.matmul(x.transpose(0, 1), y)

    # equation (14)
    u, s, vh = torch.svd(w)

    # equation (20)
    ide = torch.eye(3, device=device).double() # !!! MUST use double in GPU !!
    ide[2, 2] *= torch.det(vh @ u.transpose(0, 1))
    R = vh @ ide @ u.transpose(0, 1)

    # compute equation (4)
    t = cy - torch.matmul(R, cx)

    return R, t
    

def solve_rigid_matrix_batch(X, Y):
    """
    Solve Y = RX + t, batch processing (for a whole batch)
    Reference: https://zhuanlan.zhihu.com/p/111322916

    Inputs:
        X: BxNx3 torch tensor, where B is the batch size and N is the number of points
        Y: BxNx3 torch tensor, where B is the batch size and N is the number of points
    Returns:
        R: Bx3x3 torch tensor describing camera orientation for each batch
        t: Bx3 torch tensor describing camera translation for each batch
    """
    device = X.device
    # equation (2)
    B, N = X.shape[:2]
    cx = X.sum(dim=1) / N
    cy = Y.sum(dim=1) / N

    # equation (6)
    x = X - cx.unsqueeze(1)
    y = Y - cy.unsqueeze(1)

    # equation (13)
    w = torch.matmul(x.transpose(1, 2), y)

    # equation (13)
    u, s, vh = torch.svd(w)

    # equation (20)
    ide = torch.eye(3, device=device).repeat(B, 1, 1)#.double()
    det_R = torch.det(torch.matmul(vh, u.transpose(1, 2)))
    ide[:, 2, 2] *= det_R.squeeze()
    R = torch.matmul(torch.matmul(vh, ide), u.transpose(1, 2))

    # equation (4)
    t = cy - torch.einsum('bij,bj->bi', R, cx)
    return R, t



def rot_mat_error(r_gt, r_est):
    '''
    Input:
        r_gt: 3x3 GT Rotation Matrix
        r_est: 3x3 Pred Rotation Matrix
    Reference: https://blog.csdn.net/qq_32815807/article/details/115381825

    '''
    r_gt = r_gt.double()
    r_est = r_est.double()
    # 计算角度差的绝对值
    inner_value = (torch.trace(torch.matmul(torch.inverse(r_gt), r_est)) - 1) / 2
    # 确保值在[-1, 1]内，避免NaN
    inner_value = torch.clamp(inner_value, -1, 1)
    dis = torch.abs(torch.acos(inner_value))
    # 将弧度转换为度
    return dis * 180 / torch.tensor(math.pi, dtype=torch.double)



def set_seed(seed, gl_seed=0):
    """  set random seed, gl_seed used in worker_init_fn function """
    if seed >=0 and gl_seed>=0:
        seed += gl_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    ''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
    if seed >=0 and gl_seed>=0:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True






id_color_dict = {
    '3': 'Orchid',
    '4': 'Firebrick',
    '5': 'SpringGreen',
    '6': 'BlueViolet',
    '7': 'RoyalBlue',
    '8': 'MidnightBlue',
    '9': 'Deepskyblue',
    '10': 'Purple',
    
    '11': 'Blue',
    '12': 'Magenta',
    '13': 'MediumBlue',
    '15': 'DarkViolet',
    '16': 'DarkTurquoise',
    '17': 'DeepPink',
    '18': 'SaddleBrown',
    '19': 'Cyan',

    '20': 'Aliceblue',
    '21': 'SlateGrey',
    '22': 'Maroon',
    '23': 'ForestGreen',
    '24': 'Sienna',
    '25': 'DarkSlateGray',
    '26': 'Orangered',
    '27': 'Red',

    '28': 'Green',
    '29': 'Brown',
    '30': 'DarkCyan',
    '31': 'Lime',
    '32': 'Darkblue',
    '33': 'Chocolate',
    '34': 'DarkGreen',
    '35': 'Tan'
}


def generate_point_cloud_cluster(center, radius, num_points):
    # 生成球坐标
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)

    # 转换为笛卡尔坐标
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    return np.column_stack((x, y, z))


def visualize_pred_landmark(
    landmark_ids,
    pred_idx,
    sample_points,
    scale,
    center,
    ori_points,
    faces
    ):

    landmark_vertices = []
    landmark_colors = []
    for ldm_i, ldm_id in enumerate(landmark_ids):
        pred_idx_i = pred_idx[ldm_i]
        pred_coord_i = sample_points[pred_idx_i]   # np(3), (x, y, z)
        pred_coord_i = pred_coord_i * scale + center  # np(3), (x, y, z), original size
        x, y, z = float(pred_coord_i[0]), float(pred_coord_i[1]), float(pred_coord_i[2])

        ldm_cluster = generate_point_cloud_cluster(center=(x, y, z), radius=0.3, num_points=1000)
        color = matplotlib.colors.to_rgb(id_color_dict[ldm_id])
        color = np.repeat([color], 1000, axis=0)

        landmark_colors.append(color)
        landmark_vertices.append(ldm_cluster)
    landmark_vertices = np.concatenate(landmark_vertices, axis=0)
    landmark_colors = np.concatenate(landmark_colors, axis=0)
    landmark = trimesh.points.PointCloud(vertices=landmark_vertices, colors=landmark_colors)
    tooth = trimesh.Trimesh(vertices=ori_points, faces=faces)

    return landmark, tooth
