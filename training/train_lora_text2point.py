#Create: backend/train/train_lora_text2point.py
#!/usr/bin/env python3
import os, json, glob, math, time, random, numpy as np, torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.models.download import load_checkpoint
from point_e.diffusion import gaussian_diffusion

# your LoRA helpers
from backend.lora.inject import inject_lora, lora_params

CACHE_DIR = Path("backend/data/cap3d_cache")
RUN_DIR   = Path(os.environ.get("RUN_DIR", "backend/runs/lora-textvec"))
RUN_DIR.mkdir(parents=True, exist_ok=True)

class NPZDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(str(Path(root) / "*.npz")))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        xyz = d["xyz"].astype(np.float32)          # [N,3] in unit sphere
        rgb = d["rgb"].astype(np.uint8)            # [N,3] 0..255
        cap = str(d["caption"])
        # pack channels R,G,B to 0..1 floats if you want to condition color learning
        return {"xyz": xyz, "rgb": rgb, "caption": cap}

def collate(batch):
    # stack into tensors
    xyz = torch.from_numpy(np.stack([b["xyz"] for b in batch], 0))   # [B,N,3]
    rgb = torch.from_numpy(np.stack([b["rgb"] for b in batch], 0))   # [B,N,3]
    caps = [b["caption"] for b in batch]
    return {"xyz": xyz, "rgb": rgb, "caption": caps}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load Point·E text-conditioned base
    base_name = "base40M-textvec"
    model = model_from_config(MODEL_CONFIGS[base_name], device)
    model.load_state_dict(load_checkpoint(base_name, device))
    model.eval()

    # 2) Diffusion config + training mode
    diff = diffusion_from_config(DIFFUSION_CONFIGS[base_name])  # has training_losses()

    # 3) Inject LoRA (wrap Linear layers in attn/MLP)
    model = inject_lora(model, r=16, alpha=16, dropout=0.0, verbose=True)
    # Freeze everything except LoRA params
    for p in model.parameters(): p.requires_grad = False
    for p in lora_params(model): p.requires_grad = True

    # 4) Data
    ds = NPZDataset(CACHE_DIR)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)

    # 5) Optim
    opt = optim.AdamW(lora_params(model), lr=5e-4, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 6) Train loop (denoising objective)
    model.train()
    step=0; t0=time.time()
    for epoch in range(1):  # start with 1 epoch; you can extend later
        for batch in dl:
            xyz = batch["xyz"].to(device)           # [B,N,3]
            # pack RGB into aux channels if your model supports it
            texts = batch["caption"]                # list of strings

            # Build kwargs the model expects
            kwargs = dict(texts=texts)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                # training_losses returns loss dict for current t sampled inside
                losses = diff.training_losses(model, xyz, model_kwargs=kwargs)
                loss = losses["loss"].mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            step += 1
            if step % 25 == 0:
                dt = time.time()-t0
                print(f"step {step}  loss {loss.item():.4f}  {dt/25:.3f}s/step")
                t0=time.time()

            if step % 500 == 0:
                # Save LoRA adapters only (keep small)
                ckpt = RUN_DIR / f"lora_step{step}.pt"
                torch.save({k:v for k,v in model.state_dict().items() if "lora_" in k}, ckpt)
                print("saved", ckpt)

    # final save
    ckpt = RUN_DIR / "lora_final.pt"
    torch.save({k:v for k,v in model.state_dict().items() if "lora_" in k}, ckpt)
    print("saved", ckpt)

if __name__ == "__main__":
    main()
