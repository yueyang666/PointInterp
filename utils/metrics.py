import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance_kdtree(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    使用 KDTree 加速計算兩個點雲之間的 Chamfer Distance（平方距離版本）。

    Chamfer Distance 定義為：
        CD(P, G) = mean_{g in G} min_{p in P} ||g - p||^2
                 + mean_{p in P} min_{g in G} ||p - g||^2

    Args:
        pred (np.ndarray): 預測點雲，形狀 (M, 3)，每一列為一個 3D 點 (x, y, z)。
        gt   (np.ndarray): 真實點雲 (ground truth)，形狀 (N, 3)。

    Returns:
        float: 計算出的 Chamfer Distance，等於兩側最小平方距離的平均和。
    """
    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)

    dist_pred_to_gt, _ = tree_pred.query(gt)  # 每個 gt 找最近的 pred
    dist_gt_to_pred, _ = tree_gt.query(pred)  # 每個 pred 找最近的 gt

    loss = np.mean(dist_pred_to_gt**2) + np.mean(dist_gt_to_pred**2)
    return loss

def get_delta(orig: np.ndarray, tgt: np.ndarray, thr=0.05) -> np.ndarray:
    """
    計算 tgt 相對於 orig 的差集 (delta)，
    只保留那些到 orig 最近距離 > thr 的 tgt 點。

    Args:
        orig (np.ndarray): 原始點雲 (N,3)
        tgt  (np.ndarray): 目標點雲 (M,3)
        thr  (float)     : 最大視為相同點的距離閾值

    Returns:
        np.ndarray: 差集點雲 (K,3)
    """
    tree_o = cKDTree(orig)
    d2, _ = tree_o.query(tgt, k=1)
    diff_mask = d2 > thr
    delta = tgt[diff_mask]
    return delta