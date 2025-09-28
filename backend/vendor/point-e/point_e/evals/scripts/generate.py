#!/usr/bin/env python3
import os, numpy as np, torch
from plyfile import PlyData, PlyElement

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def write_ply(path, coords_f32, colors_u8):
    assert coords_f32.shape[1] == 3 and colors_u8.shape[1] == 3
    import numpy as np
    verts = np.empty(coords_f32.shape[0], dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('red','u1'),('green','u1'),('blue','u1')
    ])
    verts['x'] = coords_f32[:,0]
    verts['y'] = coords_f32[:,1]
    verts['z'] = coords_f32[:,2]
    verts['red']   = colors_u8[:,0]
    verts['green'] = colors_u8[:,1]
    verts['blue']  = colors_u8[:,2]
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=True).write(path)

def main():
    prompt = os.environ.get("POINT_E_PROMPT", "a shiny red sports car")
    out_dir = os.environ.get("POINT_E_OUT", "backend/data/outputs/pointclouds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.ply")

    device = get_device()
    print("Using device:", device)
    print("Prompt:", prompt)

    base_name      = "base40M-textvec"   # text-conditioned base
    upsampler_name = "upsample"

    base_model = model_from_config(MODEL_CONFIGS[base_name], device); base_model.eval()
    base_diff  = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_model.load_state_dict(load_checkpoint(base_name, device))

    up_model = model_from_config(MODEL_CONFIGS[upsampler_name], device); up_model.eval()
    up_diff  = diffusion_from_config(DIFFUSION_CONFIGS[upsampler_name])
    up_model.load_state_dict(load_checkpoint(upsampler_name, device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, up_model],
        diffusions=[base_diff, up_diff],
        num_points=[1024, 4096-1024],
        aux_channels=['R','G','B'],
        guidance_scale=[3.0, 0.0],      # 2 values (one per stage)
        use_karras=[True, True],        # 2 values (one per stage)
        model_kwargs_key_filter=('texts', '')
    )

    last = None
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
        last = x

    pc = sampler.output_to_point_clouds(last)[0]
    coords = pc.coords.astype(np.float32)
    #import numpy as np
    colors = np.stack([pc.channels['R'], pc.channels['G'], pc.channels['B']], axis=1).astype(np.uint8)

    write_ply(out_path, coords, colors)
    print("Saved:", out_path, "points:", len(coords))

if __name__ == "__main__":
    main()
