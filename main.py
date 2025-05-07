import numpy as np
import open3d as o3d
import random as rd
import time
import torch
from pointnet_sr_mini import PointNetSRMini  # 確保模型檔存在
from utils.metrics import chamfer_distance_kdtree, get_delta
from utils.io import load_point_clouds


# --- 載入模型 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")

model_32to64 = PointNetSRMini().to(device)
model_32to64.load_state_dict(torch.load('ckpt_srmini/best_pointnet_sr_32to64.pth'))
model_32to64.eval()


# --- 載入點雲 
t0 = time.time()
pc16_list, pc32_list, pc64_list = load_point_clouds('./dataset/single_sample_frame.h5') # single_sample_frame train_sample148 verifty340
print(f"讀取時間：{time.time() - t0:.4f} s")
print(f"資料集點雲數量: {len(pc16_list)}")

frame = rd.randint(0,len(pc16_list)-1)
print(f"隨機選擇第 {frame} 幀")
pc16 = pc16_list[frame][:, :3]
pc32 = pc32_list[frame][:, :3]
pc64 = pc64_list[frame][:, :3]
delta = get_delta(pc32, pc64)

source = pc16.copy()
target = pc32.copy()

# 插成64線
# pc_interp = eval.interpolate_lidar_rings(source, original_num_rings=16, target_num_rings=32, distance_thresh = 1.5)
# loss = eval.chamfer_distance_kdtree(pc_interp, target)

delta_tensor = torch.from_numpy(delta).unsqueeze(0).float().to(device)
with torch.no_grad():
    pred = model_32to64(delta_tensor).squeeze(0).cpu().numpy()
merged = np.vstack([pred, pc32])
loss = chamfer_distance_kdtree(merged, pc64)

print(f'原始幀數量{source.shape[0]}')
print(f'插值幀數量{merged.shape[0]}')
print(f'目標幀數量{target.shape[0]}')
print(f"第 {frame} 幀 Chamfer Distance: {loss:.6f}")

original = o3d.geometry.PointCloud()
original.points = o3d.utility.Vector3dVector(source)
original.paint_uniform_color([0, 0, 1]) # 藍色

pred = o3d.geometry.PointCloud()
pred.points = o3d.utility.Vector3dVector(merged)
pred.paint_uniform_color([1, 0, 0]) # 紅色

gt = o3d.geometry.PointCloud()
gt.points = o3d.utility.Vector3dVector(target)
gt.paint_uniform_color([0.0, 0.75, 0.75]) # 青色

o3d.visualization.draw_geometries([gt, original])
o3d.visualization.draw_geometries([pred, gt])

# 可視化點雲
# original = o3d.geometry.PointCloud()
# original.points = o3d.utility.Vector3dVector(source)
# original.paint_uniform_color([0, 0, 1]) # 藍色

# pred = o3d.geometry.PointCloud()
# pred.points = o3d.utility.Vector3dVector(pc_interp)
# pred.paint_uniform_color([1, 0, 0]) # 紅色

# gt = o3d.geometry.PointCloud()
# gt.points = o3d.utility.Vector3dVector(target)
# gt.paint_uniform_color([0.0, 0.75, 0.75]) # 青色

# o3d.visualization.draw_geometries([pred, original])
# o3d.visualization.draw_geometries([pred, gt])
