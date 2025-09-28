#!/usr/bin/env python3
import os, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config

from lora.inject import inject_lora, lora_params

def read_ply(path):
    import plyfile
    with open(path, "rb") as f:
        ply = plyfile.PlyData.read(f)
    verts = np.stack([ply["vertex"][a] for a in ("x","y","z")],1).astype(np.float32)
    rgb = None
    for k in ("red","green","blue","r","g","b"):
        if k not in ply["vertex"].data.dtype.names: rgb=None; break
        # if any missing, rgb=None; but we’ll break gracefully below
    names = ply["vertex"].data.dtype.names
    if all(c in names for c in ("red","green","blue")):
        rgb = np.stack([ply["vertex"][c] for c in ("red","green","blue")],1).astype(np.float32)/255.0
    elif all(c in names for c in ("r","g","b")):
        rgb = np.stack([ply["vertex"][c] for c in ("r","g","b")],1).astype(np.float32)/255.0
    else:
        rgb = np.ones_like(verts) * 0.5
    return verts, rgb

def normalize_unit_sphere(xyz):
    c = xyz.mean(0, keepdims=True)
    xyz0 = xyz - c
    scale = np.max(np.linalg.norm(xyz0, axis=1))
    if scale < 1e-8: scale = 1.0
    return xyz0 / scale

def fps_downsample(xyz, n):
    # light-weight fallback — random subset. Replace with true FPS if you like.
    if len(xyz) <= n: return np.arange(len(xyz))
    idx = np.random.choice(len(xyz), size=n, replace=False)
    return idx

class PairPLYDataset(Dataset):
    def __init__(self, manifest, n_low=1024, n_high=4096):
        import pandas as pd
        self.df = pd.read_csv(manifest)
        self.n_low, self.n_high = n_low, n_high
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        path = self.df.iloc[i]["ply_path"]
        xyz, rgb = read_ply(path)
        xyz = normalize_unit_sphere(xyz)
        # --- sample high-res target ---
        ih = fps_downsample(xyz, self.n_high)
        xyz_h, rgb_h = xyz[ih], rgb[ih]
        # --- sample low-res input from the same cloud ---
        il = fps_downsample(xyz_h, self.n_low)
        xyz_l, rgb_l = xyz_h[il], rgb_h[il]
        # pack to (C, N)
        high = np.concatenate([xyz_h, rgb_h],1).T.astype(np.float32)  # (6, N_high)
        low  = np.concatenate([xyz_l, rgb_l],1).T.astype(np.float32)  # (6, N_low)
        return torch.from_numpy(high), torch.from_numpy(low)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    device = get_device()
    print("Device:", device)

    FT_MANIFEST = os.environ.get("FT_MANIFEST", "/workspace/backend/data/objaverse-xl/train_manifest.csv")
    FT_VAL_MANIFEST = os.environ.get("FT_VAL_MANIFEST", "")
    OUTDIR = os.environ.get("FT_OUTDIR", "/workspace/backend/runs/lora-point-e-up")
    N_LOW  = int(os.environ.get("FT_LOW", "1024"))
    N_HIGH = int(os.environ.get("FT_HIGH", "4096"))
    BATCH  = int(os.environ.get("FT_BATCH", "2"))
    EPOCHS = int(os.environ.get("FT_EPOCHS", "1"))
    LR     = float(os.environ.get("FT_LR", "1e-4"))
    WORKERS= int(os.environ.get("FT_WORKERS", "4"))
    R      = int(os.environ.get("FT_LORA_R", "32"))
    ALPHA  = int(os.environ.get("FT_LORA_ALPHA", "64"))
    DROPOUT= float(os.environ.get("FT_DROPOUT", "0.05"))

    os.makedirs(OUTDIR, exist_ok=True)

    # --- build upsampler model + diffusion ---
    up_name = "upsample"
    up_model = model_from_config(MODEL_CONFIGS[up_name], device)
    up_model.load_state_dict(load_checkpoint(up_name, device))
    up_model.eval()

    # inject LoRA
    up_model = inject_lora(up_model, r=R, alpha=ALPHA, dropout=DROPOUT, verbose=True)

    # freeze base weights
    for n,p in up_model.named_parameters(): p.requires_grad = False
    for n,p in up_model.named_parameters():
        if "lora_A" in n or "lora_B" in n: p.requires_grad = True

    trainable = [(n,p.numel()) for n,p in up_model.named_parameters() if "lora_" in n and p.requires_grad]
    print("Trainable LoRA tensors:", len(trainable), "total params:", sum(x[1] for x in trainable))

    up_diff = diffusion_from_config(DIFFUSION_CONFIGS[up_name])

    # data
    train_ds = PairPLYDataset(FT_MANIFEST, n_low=N_LOW, n_high=N_HIGH)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                          pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(lora_params(up_model), lr=LR, weight_decay=0.01)

    for ep in range(1, EPOCHS+1):
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{EPOCHS}")
        for high, low in pbar:
            # shapes: high (B,6,N_high), low (B,6,N_low)
            high = high.to(device, non_blocking=True)
            low  = low.to(device, non_blocking=True)

            t = torch.randint(0, up_diff.num_timesteps, (high.size(0),), device=device)

            loss_dict = up_diff.training_losses(up_model, high, t, model_kwargs={"low_res": low})
            loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params(up_model), 1.0)
            opt.step()

            pbar.set_postfix(loss=float(loss))

        torch.save({k:v for k,v in up_model.state_dict().items() if "lora_" in k},
                   os.path.join(OUTDIR, "lora_up_ep_last.pt"))
        print(f"[ep {ep}] saved -> {os.path.join(OUTDIR,'lora_up_ep_last.pt')}")

if __name__ == "__main__":
    main()
