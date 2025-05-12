import h5py
import numpy as np
from typing import Tuple, List


def load_point_clouds(h5_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    讀取 HDF5 中的 16/32/64 線點雲
    回傳三個 list[np.ndarray]
    """
    pc16, pc32, pc64 = [], [], []
    with h5py.File(h5_path, 'r') as f:
        for ring in ('16','32','64'):
            if ring not in f:
                raise KeyError(f"Missing group {ring} in {h5_path}")
        scenes = f['16'].keys()
        for s in scenes:
            pc16.append(np.array(f['16'][s]))
            pc32.append(np.array(f['32'][s]))
            pc64.append(np.array(f['64'][s]))
    return pc16, pc32, pc64

def save_multi_to_h5(output_path: str, pcs_16: List[np.ndarray], pcs_32: List[np.ndarray], pcs_full: List[np.ndarray]):
    """
    把三種點雲一起儲存到 h5
    """
    with h5py.File(output_path, 'w') as f:
        grp16 = f.create_group('16')
        grp32 = f.create_group('32')
        grpfull = f.create_group('64')

        for idx in range(len(pcs_full)):
            grp16.create_dataset(str(idx), data=pcs_16[idx], compression='gzip')
            grp32.create_dataset(str(idx), data=pcs_32[idx], compression='gzip')
            grpfull.create_dataset(str(idx), data=pcs_full[idx], compression='gzip')
