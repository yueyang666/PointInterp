# kitti_to_multiring_h5.py

import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
from typing import List
from utils.io import save_multi_to_h5

def load_kitti_bin(bin_file: str) -> np.ndarray:
    """
    讀取單個 KITTI .bin 檔，輸出 (N,4) ndarray：x, y, z, intensity
    """
    return np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)


def split_by_ring(
    pc: np.ndarray,
    orig_rings: int = 64,
    tgt_rings: int = 32
) -> np.ndarray:
    """
    根據仰角將點雲分成 orig_rings 環，再選出 tgt_rings 環作為子集。
    """
    # 1) 計算每點仰角
    xyz = pc[:, :3]
    elev = np.arcsin(xyz[:, 2] / np.linalg.norm(xyz, axis=1))
    # 2) 分箱
    bins = np.linspace(elev.min(), elev.max(), orig_rings + 1)
    ring_idx = np.digitize(elev, bins) - 1
    ring_idx = np.clip(ring_idx, 0, orig_rings - 1)
    # 3) 選出目標環
    selected = np.linspace(0, orig_rings-1, tgt_rings, dtype=int)
    mask = np.isin(ring_idx, selected)
    return pc[mask]


def collect_bin_files(bin_patterns: List[str], max_files: int=None) -> List[str]:
    """
    根據一或多個 glob pattern 收集 .bin 檔案，並依序排序、截斷。
    """
    files = []
    for pattern in bin_patterns:
        files.extend(sorted(glob.glob(pattern)))
    if max_files:
        files = files[:max_files]
    return files


def kitti_multiring_to_h5(
    bin_patterns: List[str],
    output_h5: str,
    orig_rings: int = 64,
    max_files: int = None):

    """
    將多條 KITTI 資料路徑下的 .bin 檔轉成
    16/32/64 三組 ring 並存入同一 HDF5。
    """
    bin_files = collect_bin_files(bin_patterns, max_files)
    print(f"[INFO] 找到 {len(bin_files)} 個 .bin 檔案，開始處理...")

    pcs_16, pcs_32, pcs_full = [], [], []

    for bin_path in tqdm(bin_files, desc="讀取 & 分環"):
        pc = load_kitti_bin(bin_path)
        pcs_full.append(pc)
        pcs_32.append(split_by_ring(pc, orig_rings, 32))
        pcs_16.append(split_by_ring(pc, orig_rings, 16))

    # 使用 utils.io 裡的儲存函式
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    save_multi_to_h5(output_h5, pcs_16, pcs_32, pcs_full)
    print(f"[INFO] 已儲存至 {output_h5}")


if __name__ == "__main__":
    # 1) 定義你的 glob 路徑列表
    bin_patterns = [
        "/media/alex/Data/KITTI_Dataset/Road/2011_09_26_drive_0032_sync/velodyne_points/data/*.bin",
        "/media/alex/Data/KITTI_Dataset/Road/2011_09_30_drive_0016_sync/velodyne_points/data/*.bin",
        "/media/alex/Data/KITTI_Dataset/Residential/2011_09_26_drive_0022_sync/velodyne_points/data/*.bin",
        # ... 更多 pattern ...
    ]

    # 2) 輸出 H5 路徑
    output_path = "./dataset/verify_data.h5"

    # 3) 呼叫主函式
    kitti_multiring_to_h5(
        bin_patterns,
        output_path,
        orig_rings=64,
        max_files=None
    )
