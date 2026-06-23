"""
展示单个 PLY — 用 2DGS MiniCam (已验证渲染稀疏 surfer 不崩溃)
用法: python scripts/utils/render_ply.py --ply <path> --output <png> --views 8
"""
import os, sys, argparse, numpy as np, torch, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '2d-gaussian-splatting'))
from gaussian_renderer import GaussianModel, render
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--output', default='output/fusion/render.png')
    ap.add_argument('--views', type=int, default=8)
    ap.add_argument('--res', type=int, default=600)
    args = ap.parse_args()

    gs = GaussianModel(3); gs.load_ply(args.ply)
    xyz = gs.get_xyz.detach().cpu().numpy()
    c = np.array([xyz[:,0].mean(), xyz[:,1].mean(), xyz[:,2].mean()])
    ext = max(xyz[:,0].max()-xyz[:,0].min(), xyz[:,1].max()-xyz[:,1].min(), xyz[:,2].max()-xyz[:,2].min())
    radius = ext * 3.5
    print(f"{os.path.basename(args.ply)}: {len(xyz)} surfels, extent={ext:.2f}, radius={radius:.1f}")

    fov = 60 * np.pi / 180; zn, zf = 0.01, 200.0
    proj = torch.tensor(getProjectionMatrix(zn, zf, fov, fov).T, dtype=torch.float32, device='cuda')
    class Pipe: convert_SHs_python= False; compute_cov3D_python= False; depth_ratio= 0.0; debug= False
    bg = torch.tensor([0.02, 0.02, 0.04], dtype=torch.float32, device='cuda')

    frames = []
    for i in range(args.views):
        angle = 2 * np.pi * i / args.views
        eye = np.array([c[0]+radius*np.cos(angle), c[1]+radius*0.3*np.sin(angle*2), c[2]+radius*np.sin(angle)])
        look = c - eye; look = look / np.linalg.norm(look)
        up = np.array([0., 1., 0.])
        cr = np.cross(up, look); cr = cr / np.linalg.norm(cr)
        cu = np.cross(look, cr)
        R = np.stack([cr, cu, look], axis=0)
        T = -R @ eye
        w2c_t = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0).T, dtype=torch.float32, device='cuda')
        cam = MiniCam(args.res, args.res, fov, fov, zn, zf, w2c_t,
                      w2c_t.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0))
        with torch.no_grad():
            out = render(cam, gs, Pipe(), bg)
            arr = (torch.clamp(out['render'], 0., 1.).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        if i == 0:
            print(f"  max={out['render'].max().item():.2f} alpha={out['rend_alpha'].max().item():.2f}")
        frames.append(arr)

    row = np.concatenate(frames, axis=1)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    print(f"saved: {args.output}")

if __name__ == '__main__':
    main()
