# -*- coding: utf-8 -*-
"""
PointNet-SR-mini (Full Optimized Version)

特點:
- 支援 resume from checkpoint
- 固定 random seed
- 自適應 ReduceLROnPlateau
- EarlyStopping
- 自動調整 DataLoader num_workers
- 訓練後繪製 loss 曲線
"""

import os, random, argparse, math, h5py, time, json
from pathlib import Path
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree

# -------------------------------------------------
# 0. 固定 random seed
# -------------------------------------------------
def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")

# -------------------------------------------------
# 1. Utilities
# -------------------------------------------------
def chamfer_distance(p_pred: torch.Tensor, p_gt: torch.Tensor) -> torch.Tensor:
    diff = p_pred.unsqueeze(2) - p_gt.unsqueeze(1)  # (B,N,M,3)
    dist2 = (diff ** 2).sum(-1)
    min_pred2gt,_ = dist2.min(dim=2)
    min_gt2pred,_ = dist2.min(dim=1)
    cd = min_pred2gt.mean(dim=1) + min_gt2pred.mean(dim=1)
    return cd.mean()

# -------------------------------------------------
# 2. Dataset Loader
# -------------------------------------------------
class SRDataset(Dataset):
    def __init__(self, h5_path: str, orig_ring: int = 16, tgt_ring: int = 32, n_in: int = 2048, n_gt: int = 4096):
        super().__init__()
        self.n_in, self.n_gt = n_in, n_gt
        self.orig_ring, self.tgt_ring = orig_ring, tgt_ring
        self.frames = self._load_h5(h5_path)

    def _load_h5(self, h5_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        with h5py.File(h5_path, 'r') as f:
            p_orig = list(f[str(self.orig_ring)].values())
            p_tgt  = list(f[str(self.tgt_ring)].values())
            frames = [(np.asarray(o[:], dtype=np.float32), np.asarray(t[:], dtype=np.float32))
                      for o,t in zip(p_orig, p_tgt)]
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def _get_delta_and_gt(self, orig: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tree_t = cKDTree(tgt)
        d,_ = tree_t.query(orig, k=1)
        thr = 0.05
        inter_mask = d < thr
        gt = orig[inter_mask]
        tree_o = cKDTree(orig)
        d2,_ = tree_o.query(tgt, k=1)
        diff_mask = d2 > thr
        delta = tgt[diff_mask]
        return delta, gt

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        orig, tgt = self.frames[idx]
        delta, gt = self._get_delta_and_gt(orig, tgt)
        rng = np.random.default_rng()
        delta = delta[rng.choice(len(delta), self.n_in, replace=len(delta)<self.n_in)]
        gt    = gt[rng.choice(len(gt), self.n_gt, replace=len(gt)<self.n_gt)]
        return torch.from_numpy(delta[:, :3]), torch.from_numpy(gt[:, :3])

# -------------------------------------------------
# 3. Network
# -------------------------------------------------
class PointNetSRMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1))
        self.fc_global = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv1d(1280, 512, 1), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 3, 1))

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x = pts.transpose(1, 2)
        feat = self.mlp1(x)
        global_feat = torch.max(feat, 2)[0]
        g = self.fc_global(global_feat).unsqueeze(-1).repeat(1, 1, pts.size(1))
        cat = torch.cat([feat, g], dim=1)
        offset = self.decoder(cat).transpose(1, 2)
        return pts + offset

# -------------------------------------------------
# 4. Trainer
# -------------------------------------------------
def train(args: argparse.Namespace):
    if torch.cuda.is_available():
        print(f"[INFO] System find GPU")
    else:
        print(f"[INFO] System use CPU")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    num_workers = min(8, os.cpu_count() // 2)
    ds_train = SRDataset(args.train_h5, orig_ring=args.orig, tgt_ring=args.tgt)
    dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=num_workers, drop_last=True)

    ds_val, dl_val = None, None
    if args.val_h5 and Path(args.val_h5).exists():
        ds_val = SRDataset(args.val_h5, orig_ring=args.orig, tgt_ring=args.tgt)
        dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=num_workers)
        print(f"[INFO] Validation set loaded: {len(ds_val)} samples")

    model = PointNetSRMini().to(device)
    if args.ckpt and Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt))
        print(f"[INFO] Loaded checkpoint from {args.ckpt}")

    time.sleep(3.0)
    print(f"[INFO] System start training!")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    history = {"train_loss":[], "val_loss":[]}
    early_stop_counter = 0
    early_stop_patience = 50

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0
        for delta, gt in tqdm(dl_train, desc=f"Epoch {epoch:03d}"):
            delta, gt = delta.to(device), gt.to(device)
            pred = model(delta)
            loss = chamfer_distance(pred, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
        train_loss /= len(dl_train)
        print(f"Epoch {epoch:03d} Train Loss: {train_loss:.6f}")
        history["train_loss"].append(train_loss)

        # Validation
        val_loss = None
        if dl_val:
            model.eval(); val_loss = 0
            with torch.no_grad():
                for delta, gt in dl_val:
                    delta, gt = delta.to(device), gt.to(device)
                    pred = model(delta)
                    val_loss += chamfer_distance(pred, gt).item()
            val_loss /= len(dl_val)
            print(f"              Val Loss: {val_loss:.6f}")
            history["val_loss"].append(val_loss)

            # Adjust learning rate
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), out/f'best_pointnet_sr_{args.orig}to{args.tgt}.pth')
                print("[INFO] Best model updated!")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"[INFO] EarlyStopping counter: {early_stop_counter}/{early_stop_patience}")

            if early_stop_counter >= early_stop_patience:
                print("[INFO] EarlyStopping triggered")
                break

    # Save loss history
    with open(out/'loss_history.json', 'w') as f:
        json.dump(history, f)
    print("[INFO] Training finished.")

    # Plot
    plt.plot(history["train_loss"], label='Train Loss')
    if history["val_loss"]:
        plt.plot(history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(out/'loss_curve.png')
    print("[INFO] Save loss_curve.png")
    plt.show()

# -------------------------------------------------
# 5. CLI
# -------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_h5', required=True)
    ap.add_argument('--val_h5', default=None)
    ap.add_argument('--ckpt', default=None, help='(Optional) path to pretrained checkpoint')
    ap.add_argument('--orig', type=int, default=16)
    ap.add_argument('--tgt', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--out', default='ckpt_srmini')
    args = ap.parse_args()
    train(args)
