import numpy as np
import torch


def euclidean_distance(qf, gf):
    """
    计算 query-gallery 欧氏距离矩阵
    qf: [m, d]
    gf: [n, d]
    return: [m, n]
    """
    m, n = qf.size(0), gf.size(0)
    dist_mat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_distance(qf, gf):
    """
    计算 query-gallery 余弦距离矩阵
    qf: [m, d]
    gf: [n, d]
    return: [m, n]
    """
    qf = torch.nn.functional.normalize(qf, dim=1, p=2)
    gf = torch.nn.functional.normalize(gf, dim=1, p=2)
    dist_mat = 1 - torch.mm(qf, gf.t())
    return dist_mat.cpu().numpy()


def re_ranking(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    """
    CVPR2017 re-ranking: k-reciprocal encoding
    probFea: query features, shape [num_query, feat_dim]
    galFea: gallery features, shape [num_gallery, feat_dim]
    return: final_dist, shape [num_query, num_gallery]
    """
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)

    feat = torch.cat([probFea, galFea], dim=0)

    # 计算全体样本两两欧氏距离
    distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
    distmat = distmat + distmat.t()
    distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
    original_dist = distmat.cpu().numpy()
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)

    initial_rank = np.argsort(original_dist).astype(np.int32)

    # k-reciprocal neighbors
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]

        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    # query expansion
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    del initial_rank

    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]

        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] += np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist[:query_num, :] * lambda_value
    final_dist = final_dist[:, query_num:]

    return final_dist