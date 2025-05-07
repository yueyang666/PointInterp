import numpy as np
from scipy.spatial import cKDTree


def interpolate_lidar_rings(
    pc: np.ndarray,
    original_num_rings: int = 32,
    target_num_rings: int = 64,
    distance_thresh: float = 0.15,
    azimuth_bin_deg: float = 0.5,
    exact_ring: bool = True,
    check_azimuth: bool = True
) -> np.ndarray:
    """
    主函式：仰角插值 + 可選精確環數 + 可選水平方位角平衡
    """
    # 1) 仰角插值中點
    mid_pts = _angle_binning_and_midpoints(pc, original_num_rings, distance_thresh)
    merged = np.vstack([pc, mid_pts])

    # 2) 可選：精確補足 target_num_rings 環
    if exact_ring:
        # 重新計算 e_min, e_max
        r = np.linalg.norm(pc, axis=1)
        elev = np.arcsin(pc[:, 2] / r)
        e_min, e_max = elev.min(), elev.max()
        merged = _enforce_exact_rings(merged, target_num_rings, e_min, e_max)

    # 3) 可選：水平方位角檢查 & 補點
    if check_azimuth:
        merged = _azimuth_balance(merged, target_num_rings, azimuth_bin_deg)

    return merged

def _angle_binning_and_midpoints(
    pc: np.ndarray,
    original_num_rings: int,
    distance_thresh: float
) -> np.ndarray:
    """
    1. 根據仰角將點雲分箱 (原始環數)
    2. 對相鄰兩環進行 KDTree 最近鄰配對，插入中點
    Returns:
        mid_pts: 插值後的中點陣列 (K,3)
    """
    # 仰角分箱
    r = np.linalg.norm(pc, axis=1)
    elev = np.arcsin(pc[:, 2] / r)
    e_min, e_max = elev.min(), elev.max()
    bins = np.linspace(e_min, e_max, original_num_rings + 1)
    ring_idx = np.clip(np.digitize(elev, bins) - 1, 0, original_num_rings - 1)
    rings = [pc[ring_idx == i] for i in range(original_num_rings)]

    # 相鄰環插值中點
    mids = []
    for a, b in zip(rings[:-1], rings[1:]):
        if a.size == 0 or b.size == 0:
            continue
        tree_b = cKDTree(b)
        d, nn = tree_b.query(a, k=1)
        mask = d < distance_thresh
        if np.any(mask):
            mids.append(0.5 * (a[mask] + b[nn[mask]]))

    return np.vstack(mids) if mids else np.empty((0, 3))

def _enforce_exact_rings(
    merged: np.ndarray,
    target_num_rings: int,
    e_min: float,
    e_max: float
) -> np.ndarray:
    """
    重新依 target_num_rings 對 merged 點雲做仰角分箱，
    隨機抽樣或重複補點，保證每一環至少有一點、總共剛好 target_num_rings 環。
    """
    elev_all = np.arcsin(merged[:, 2] / np.linalg.norm(merged, axis=1))
    tgt_bins = np.linspace(e_min, e_max, target_num_rings + 1)
    tgt_idx = np.clip(np.digitize(elev_all, tgt_bins) - 1, 0, target_num_rings - 1)

    final_pts = []
    rng = np.random.default_rng()
    for i in range(target_num_rings):
        pts_i = merged[tgt_idx == i]
        if pts_i.size == 0:
            # 從最近有點的環複製一個
            left, right = i - 1, i + 1
            while left >= 0 or right < target_num_rings:
                cand = None
                if left >= 0 and np.any(tgt_idx == left):
                    cand = merged[tgt_idx == left]
                elif right < target_num_rings and np.any(tgt_idx == right):
                    cand = merged[tgt_idx == right]
                if cand is not None:
                    pts_i = cand[rng.integers(len(cand), size=1)]
                    break
                left -= 1; right += 1
        final_pts.append(pts_i)
    return np.vstack(final_pts)

def _azimuth_balance(
    merged: np.ndarray,
    target_num_rings: int,
    azimuth_bin_deg: float
) -> np.ndarray:
    """
    根據水平方位角 (0~2π) 將 merged 分箱，
    如果某一箱的點數小於 target_num_rings，將該箱內的點隨機重複取樣補齊。
    """
    az = np.arctan2(merged[:,1], merged[:,0])
    az = (az + 2*np.pi) % (2*np.pi)
    num_bins = int(360 / azimuth_bin_deg)
    bins_az = np.linspace(0, 2*np.pi, num_bins + 1)
    az_idx = np.clip(np.digitize(az, bins_az) - 1, 0, num_bins - 1)

    rng = np.random.default_rng()
    additions = []
    for i in range(num_bins):
        pts_i = merged[az_idx == i]
        if 0 < pts_i.shape[0] < target_num_rings:
            extra = target_num_rings - pts_i.shape[0]
            additions.append(pts_i[rng.integers(len(pts_i), size=extra)])
    if additions:
        merged = np.vstack([merged] + additions)
    return merged
