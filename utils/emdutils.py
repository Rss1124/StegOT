import cv2
import torch
import torch.nn.functional as F


def get_weight_vector(A, B):
    M = A.shape[0]
    N = B.shape[0]

    B = F.adaptive_avg_pool2d(B, [1, 1])
    B = B.repeat(1, 1, A.shape[2], A.shape[3])

    A = A.unsqueeze(1)
    B = B.unsqueeze(0)

    A = A.repeat(1, N, 1, 1, 1)
    B = B.repeat(M, 1, 1, 1, 1)

    combination = (A * B).sum(2)
    combination = combination.view(M, N, -1)
    combination = F.relu(combination) + 1e-3
    return combination


def get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv'):
    num_query = similarity_map.shape[0]
    num_proto = similarity_map.shape[1]

    total_cost = 0
    num = 0
    for i in range(num_query):
        for j in range(num_proto):
            cost, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
            similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()
            num += 1
            total_cost += cost

    return total_cost / num, similarity_map


def get_similiarity_map(proto, query):
    way = proto.shape[0]
    num_query = query.shape[0]
    query = query.view(query.shape[0], query.shape[1], -1)
    proto = proto.view(proto.shape[0], proto.shape[1], -1)

    proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
    query = query.unsqueeze(1).repeat([1, way, 1, 1])
    proto = proto.permute(0, 1, 3, 2)
    query = query.permute(0, 1, 3, 2)
    feature_size = proto.shape[-2]

    proto = proto.unsqueeze(-3)
    query = query.unsqueeze(-2)
    query = query.repeat(1, 1, 1, feature_size, 1)
    similarity_map = F.cosine_similarity(proto, query, dim=-1)

    return similarity_map


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow


def get_emd_loss(ot_l_z, l_mix):  # (1,1,8,8)
    weight1 = get_weight_vector(ot_l_z, l_mix)  # (1,1,64)
    weight2 = get_weight_vector(l_mix, ot_l_z)  # (1,1,64)

    # weight1 = ot_l_z.reshape(1,1,64)
    # weight2 = l_mix.reshape(1,1,64)

    cosine_distance_matrix = get_similiarity_map(ot_l_z, l_mix)  # (1,1,64,64)

    emd_loss, flow = get_emd_distance(cosine_distance_matrix, weight1, weight2)
    return torch.tensor(emd_loss, requires_grad=True)
