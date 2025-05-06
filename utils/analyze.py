import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm

from utils.io import load_point_clouds   
from utils.metrics import chamfer_distance_kdtree
from pointnet_sr_mini import PointNetSRMini   

def analyze_losses(loss_array, title, filename_suffix):
    loss_array = np.array(loss_array)
    average_loss = np.mean(loss_array)
    std_loss = np.std(loss_array)
    max_loss = np.max(loss_array)
    min_loss = np.min(loss_array)
    max_idx = np.argmax(loss_array)
    min_idx = np.argmin(loss_array)

    print(f"\n=== {title} 分析報告 ===")
    print(f"總Frame數量: {len(loss_array)}")
    print(f"平均 Chamfer Distance: {average_loss:.6f}")
    print(f"標準差: {std_loss:.6f}")
    print(f"最大 Chamfer Distance: {max_loss:.6f} (Frame {max_idx})")
    print(f"最小 Chamfer Distance: {min_loss:.6f} (Frame {min_idx})")

    # 找出 Top 10
    top10_max_indices = np.argsort(-loss_array)[:5]
    top10_min_indices = np.argsort(loss_array)[:5]

    # print("\n前5個 Loss 最大的 Frame:")
    # for i, idx in enumerate(top10_max_indices):
    #     print(f"{i+1}. Frame {idx}, Loss = {loss_array[idx]:.6f}")

    # print("\n前5個 Loss 最小的 Frame:")
    # for i, idx in enumerate(top10_min_indices):
    #     print(f"{i+1}. Frame {idx}, Loss = {loss_array[idx]:.6f}")

    # === 畫圖 ===
    plt.rc('font', family='Noto Sans CJK JP') #in linux
    plt.figure(figsize=(12, 7))
    plt.plot(loss_array, label="Chamfer Distance", color='blue')
    plt.scatter(max_idx, max_loss, color='red', label=f"Max {max_loss:.4f}")
    plt.scatter(min_idx, min_loss, color='green', label=f"Min {min_loss:.4f}")

    # 畫上平均值的紅線
    plt.axhline(average_loss, color='red', linestyle='--', label=f'Average {average_loss:.4f}')
    
    plt.xlabel('Frame Index')
    plt.ylabel('Chamfer Distance')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 儲存圖片
    os.makedirs('figures', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'figures/loss_curve_{timestamp}_{filename_suffix}.png'
    plt.savefig(save_path)
    plt.show()
    print(f"\n{title} 曲線圖已儲存到: {save_path}")

def analyze_DL_all(filepath):
    """
    使用兩個 PointNet-SR-mini 模型 (16➔32, 32➔64) 進行推理與 Chamfer Distance 分析。

    流程：
        - 載入已訓練好的 16to32、32to64 模型
        - 從 H5 檔載入點雲資料 (16線, 32線, 64線)
        - 對每個 frame：
            - 計算差集 delta
            - 以 delta 作為模型輸入，推理出超解析點
            - 合併原始點雲與補點，計算與 ground truth 的 Chamfer Distance
        - 統計並分析每個階段的 Chamfer Distance

    Args:
        filepath (str): H5 檔案路徑，包含 16線、32線、64線三種點雲數據。

    需要的外部檔案:
        - ckpt_srmini/best_pointnet_sr_16to32.pth
        - ckpt_srmini/best_pointnet_sr_32to64.pth

    注意:
        - 預設只使用前三維 (x,y,z)。
        - 若某些 frame 找不到差集 (delta 點為 0)，則自動跳過。
        - 輸出 loss 曲線圖與統計結果。
    """
    # --- 基本設定 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # --- 載入模型 ---
    model_16to32 = PointNetSRMini().to(device)
    model_16to32.load_state_dict(torch.load('ckpt_srmini/best_pointnet_sr_16to32.pth'))
    model_16to32.eval()

    model_32to64 = PointNetSRMini().to(device)
    model_32to64.load_state_dict(torch.load('ckpt_srmini/best_pointnet_sr_32to64.pth'))
    model_32to64.eval()

    # --- 載入點雲資料 ---
    print(f"[INFO] 讀取 H5 中的點雲...")
    t0 = time.time()
    pc16_list, pc32_list, pc64_list = load_point_clouds(filepath)
    print(f"[INFO] 讀取完成，耗時 {time.time() - t0:.2f} 秒")

    # --- 定義差集提取 ---
    from scipy.spatial import cKDTree
    def get_delta(orig: np.ndarray, tgt: np.ndarray, thr=0.05):
        tree_o = cKDTree(orig)
        d2, _ = tree_o.query(tgt, k=1)
        diff_mask = d2 > thr
        delta = tgt[diff_mask]
        return delta

    # --- 推理並計算 Chamfer Distance ---
    losses_16to32 = []
    losses_32to64 = []

    # 16 -> 32
    print("\n[INFO] 開始 16 ➔ 32 預測與評估")
    for idx in tqdm(range(len(pc16_list)), desc="16to32"):
        pc16 = pc16_list[idx][:, :3]
        pc32 = pc32_list[idx][:, :3]
        delta = get_delta(pc16, pc32)

        if delta.shape[0] == 0:
            continue  # 有些 frame 可能找不到差集

        delta_tensor = torch.from_numpy(delta).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model_16to32(delta_tensor).squeeze(0).cpu().numpy()
        merged = np.vstack([pred, pc16])
        loss = chamfer_distance_kdtree(merged, pc32)
        losses_16to32.append(loss)

    # 32 -> 64
    print("\n[INFO] 開始 32 ➔ 64 預測與評估")
    for idx in tqdm(range(len(pc32_list)), desc="32to64"):
        pc32 = pc32_list[idx][:, :3]
        pc64 = pc64_list[idx][:, :3]
        delta = get_delta(pc32, pc64)

        if delta.shape[0] == 0:
            continue

        delta_tensor = torch.from_numpy(delta).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model_32to64(delta_tensor).squeeze(0).cpu().numpy()
        merged = np.vstack([pred, pc32])
        loss = chamfer_distance_kdtree(merged, pc64)
        losses_32to64.append(loss)

    # --- 分析每組結果 ---
    if losses_16to32:
        analyze_losses(losses_16to32, "16->32 PointNet-SR-mini", "16to32")
    else:
        print("[WARN] 16->32 沒有計算出 loss")

    if losses_32to64:
        analyze_losses(losses_32to64, "32->64 PointNet-SR-mini", "32to64")
    else:
        print("[WARN] 32->64 沒有計算出 loss")

if __name__ == "__main__":
    """
    主程式入口

    功能說明：
        依據指令列參數，選擇不同的點雲超解析補點方式，進行 Chamfer Distance 分析與可視化。

    提供兩種補點選項：
        - 'dl' (預設)：使用已訓練的 PointNet-SR-mini 深度學習模型進行點雲補點與評估。
        - 'linear'：使用普通插值方法進行點雲補點與評估。（尚未實作，預留接口）

    指令列參數：
        --filepath        (str)  必填，H5檔案路徑，包含 16線 / 32線 / 64線 點雲。
        --interpolation   (str)  選填，補點方法，可選 'dl' 或 'linear'，預設為 'dl'。

    執行範例：
        # 使用 DL 模型
        python your_script.py --filepath dataset/test_sample148.h5 --interpolation dl

        # 使用普通插值
        python your_script.py --filepath dataset/test_sample148.h5 --interpolation linear

    注意事項：
        - 使用 'dl' 模式時，需確保目錄下已有訓練好的模型：
            - ckpt_srmini/best_pointnet_sr_16to32.pth
            - ckpt_srmini/best_pointnet_sr_32to64.pth
        - H5 檔案格式需符合：
            /16/xxx ➔ (N,4) 、/32/xxx ➔ (M,4) 、/64/xxx ➔ (L,4)
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True, help='H5檔案路徑 (包含16/32/64線點雲)')
    parser.add_argument('--interpolation', type=str, choices=['dl', 'linear'], default='dl',
                        help="選擇補點方法：'dl' 使用PointNet-SR-mini，'linear' 使用普通插值")
    args = parser.parse_args()

    if args.interpolation == 'dl':
        analyze_DL_all(args.filepath)
    elif args.interpolation == 'linear':
        print("[INFO] 使用普通插值進行分析 (尚未實作)")
        # TODO: 這裡可以放 linear interpolation 分析的函式
    else:
        raise ValueError(f"不支援的補點方法: {args.interpolation}")