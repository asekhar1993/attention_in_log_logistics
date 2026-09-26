"""
Plot per-expert attention heatmaps for a single WSI from a trained EGMLL checkpoint.

Mirrors the loading pattern already used in train.py / plot.py in this codebase:
  cfg -> create_WSI_model(cfg) -> load epoch_N.pt state_dict -> DataGeneratorTCGASurvivalWSIEGMLL
so preprocessing/normalization of features stays identical to what the model was trained on
(rather than re-reading the raw extract_features_fp_p.py .h5 files directly, which could drift
from whatever the DataGenerator does internally).

Usage (bare grid heatmap, no WSI image needed):
    python plot_expert_heatmaps.py --config configs/luad_sgcmll.yaml --fold 1 --epoch 20 \
        --slide_id TCGA-XX-XXXX --split val --patch_size 224 --out_dir heatmaps/

Usage (overlaid on the actual slide thumbnail via openslide):
    python plot_expert_heatmaps.py --config configs/luad_sgcmll.yaml --fold 1 --epoch 20 \
        --slide_id TCGA-XX-XXXX --split val --patch_size 224 --out_dir heatmaps/ \
        --wsi_dir /workspace/wsis/tcga/luad --slide_ext .svs
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from munch import Munch
from PIL import Image
from torch.utils.data import DataLoader

from dataset.dataset_survival_egmll import DataGeneratorTCGASurvivalWSIEGMLL
from models import create_WSI_model

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


def set_random_seed(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model(cfg, save_dir, epoch, device):
    model = create_WSI_model(cfg)
    ckpt_path = os.path.join(save_dir, 'weights', f'epoch_{epoch}.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def find_slide_batch(loader, slide_id):
    """Iterate the loader (batch_size=1) until the requested slide_id is found."""
    for batch in loader:
        data, constant_dict = batch
        if data['wid'][0] == slide_id:
            return data, constant_dict
    raise ValueError(f"slide_id '{slide_id}' not found in this split.")


def get_expert_attentions(model, data, device):
    """
    Run the forward pass and return, per expert:
        - attention values already unpermuted back to the ORIGINAL patch order
          (i.e. index i corresponds to coords[i], the same order the dataset provides)
        - the gate weight for that expert on this slide
    Also returns the original coords array (L x 2) in that same order.
    """
    x_dict = {k: v.to(device) for k, v in data.items() if k not in ['wid', 't', 'c']}
    if 'coords' not in x_dict:
        raise RuntimeError(
            "No 'coords' key found in the batch — set with_coords=True in the dataset "
            "config, coords are required to place attention values back on the slide."
        )
    coords = x_dict['coords'].squeeze(0).detach().cpu().numpy()  # (L, 2), original order

    with torch.no_grad():
        params, subloss, extras = model(**x_dict, return_interpret=True)

    patch_order = extras.get('patch_order')
    if patch_order is None:
        raise RuntimeError(
            "Model did not return patch_order — make sure the backbone is running with "
            "as_backbone=True and as_backbone.forward returns {'feat', 'patch_order'}."
        )
    patch_order = patch_order.reshape(-1).detach().cpu().numpy()  # (L,), maps AMIL position -> original patch index

    gate = extras['gate'].squeeze(0).detach().cpu().numpy()  # (E,)
    expert_atts = extras['expert_atts']  # list of E tensors, each (1, L)

    L = coords.shape[0]
    per_expert_scores = []
    for att_e in expert_atts:
        att_e = att_e.reshape(-1).detach().cpu().numpy()
        if att_e.shape[0] != patch_order.shape[0]:
            raise RuntimeError(
                f"attention length {att_e.shape[0]} != patch_order length {patch_order.shape[0]} "
                "— use_filter_branch and clustering should preserve total patch count; check the backbone."
            )
        scores_original = np.full(L, np.nan, dtype=np.float64)
        scores_original[patch_order] = att_e  # place each attention value at its original patch index
        per_expert_scores.append(scores_original)

    return per_expert_scores, gate, coords


def scores_to_grid(scores, coords, patch_size):
    """
    Bin per-patch scores onto a regular 2D grid using each patch's top-left (x, y) coordinate.
    Assumes coords are non-overlapping patch top-left pixel coordinates spaced by patch_size,
    matching CLAM-style patch extraction (create_patches_fp.py). If your coords use a different
    convention (e.g. already grid-index coordinates), pass patch_size=1.
    """
    x0, y0 = coords[:, 0].min(), coords[:, 1].min()
    gx = np.round((coords[:, 0] - x0) / patch_size).astype(int)
    gy = np.round((coords[:, 1] - y0) / patch_size).astype(int)
    canvas = np.full((gy.max() + 1, gx.max() + 1), np.nan, dtype=np.float64)
    canvas[gy, gx] = scores
    return canvas


def normalize(scores, method='rank'):
    """
    Map per-slide scores to [0, 1] for colormapping, ignoring NaN (background) cells.

    method='rank' (default): each valid patch gets its percentile rank among valid
    patches. Attention over tens of thousands of patches is heavily right-skewed
    (most patches near ~1/L, a few standouts) — percentile min-max still leaves most
    values compressed near 0 under that skew, which is why raw-value normalization
    tends to render as mostly blue with only the extremes visible. Ranking guarantees
    the full colormap is used regardless of skew, at the cost of no longer reflecting
    absolute magnitude (only relative ordering).

    method='minmax': 1st/99th-percentile-clipped min-max on the raw values — keeps
    relative magnitude information, use this if you specifically need that.
    """
    valid_mask = ~np.isnan(scores)
    valid = scores[valid_mask]
    if valid.size == 0:
        return scores

    if method == 'rank':
        order = np.argsort(np.argsort(valid))
        ranked = order / max(1, valid.size - 1)
        out = np.full_like(scores, np.nan)
        out[valid_mask] = ranked
        return out

    lo, hi = np.percentile(valid, 1), np.percentile(valid, 99)
    if hi <= lo:
        return np.zeros_like(scores)
    return np.clip((scores - lo) / (hi - lo), 0, 1)


def get_thumbnail(wsi_path, max_dim=2000):
    """
    Return (thumbnail: PIL.Image, scale_x, scale_y) where scale_* converts a
    level-0 pixel coordinate into a thumbnail pixel coordinate.
    """
    wsi = openslide.open_slide(wsi_path)
    level0_w, level0_h = wsi.level_dimensions[0]
    thumb = wsi.get_thumbnail((max_dim, max_dim))
    scale_x = thumb.width / level0_w
    scale_y = thumb.height / level0_h
    return thumb, scale_x, scale_y


def rasterize_to_thumbnail(scores, coords, patch_size, thumb_w, thumb_h, scale_x, scale_y):
    """
    Paint each patch's score as a filled rectangle at its position on the thumbnail,
    in thumbnail pixel space. NaN (background / no patch) stays NaN.
    """
    canvas = np.full((thumb_h, thumb_w), np.nan, dtype=np.float64)
    tw = max(1, int(round(patch_size * scale_x)))
    th = max(1, int(round(patch_size * scale_y)))
    for (x, y), s in zip(coords, scores):
        if np.isnan(s):
            continue
        tx = int(round(x * scale_x))
        ty = int(round(y * scale_y))
        canvas[ty:ty + th, tx:tx + tw] = s
    return canvas


def overlay_heatmap(thumb_img, canvas, cmap_name='jet', alpha=0.5):
    """Alpha-blend a normalized (0-1, NaN=background) score canvas over the slide thumbnail."""
    cmap = plt.get_cmap(cmap_name)
    valid_mask = ~np.isnan(canvas)
    normed = np.nan_to_num(canvas, nan=0.0)
    rgba = cmap(normed)
    rgba[..., 3] = np.where(valid_mask, alpha, 0.0)  # transparent wherever there's no patch
    heat_img = Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
    base = thumb_img.convert('RGBA').resize((canvas.shape[1], canvas.shape[0]))
    return Image.alpha_composite(base, heat_img)


def plot_heatmaps(images, gate, slide_id, out_dir, is_overlay=False, cmap_name='jet', thumbnail=None):
    """
    images: either a list of 2D float canvases (bare mode) or a list of PIL Images
    (already-composited overlay mode, from overlay_heatmap()).
    thumbnail: optional plain PIL Image of the slide (no heatmap), added as an extra
    first panel in the combined figure so it can be compared directly against each
    expert's overlay. Only meaningful when is_overlay=True.
    """
    E = len(images)
    n_panels = E + (1 if thumbnail is not None else 0)
    os.makedirs(out_dir, exist_ok=True)
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    panel_idx = 0
    if thumbnail is not None:
        axes[panel_idx].imshow(np.asarray(thumbnail))
        axes[panel_idx].set_title("Original WSI")
        axes[panel_idx].axis('off')
        panel_idx += 1

    for e, img in enumerate(images):
        ax = axes[panel_idx]
        if is_overlay:
            ax.imshow(np.asarray(img))
            # colorbar reflects the 0-1 normalized score scale, not the composited pixels
            sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(0, 1))
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(img, cmap=cmap_name, interpolation='nearest')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Expert {e} (gate={gate[e]:.2f})")
        ax.axis('off')
        panel_idx += 1

    for e in range(panel_idx, len(axes)):
        axes[e].axis('off')

    fig.suptitle(f"Per-expert attention — {slide_id}")
    fig.tight_layout()
    combined_path = os.path.join(out_dir, f"{slide_id}_expert_heatmaps.png")
    fig.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {combined_path}")

    # also save each expert's heatmap individually
    for e, img in enumerate(images):
        fig, ax = plt.subplots(figsize=(6, 6))
        if is_overlay:
            ax.imshow(np.asarray(img))
            sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(0, 1))
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(img, cmap=cmap_name, interpolation='nearest')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{slide_id} — Expert {e} (gate={gate[e]:.2f})")
        ax.axis('off')
        path = os.path.join(out_dir, f"{slide_id}_expert{e}.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--slide_id', required=True, help="wid as stored by the dataset, e.g. TCGA-XX-XXXX")
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--patch_size', type=int, default=224,
                         help="Spacing between patch coordinates; must match what create_patches_fp.py used.")
    parser.add_argument('--out_dir', default='heatmaps')
    parser.add_argument('--wsi_dir', default=None,
                         help="Directory containing the raw slide files (e.g. /workspace/wsis/tcga/luad). "
                              "If set, heatmaps are overlaid on the slide thumbnail via openslide; "
                              "if omitted, a bare coordinate-grid heatmap is plotted instead.")
    parser.add_argument('--slide_ext', default='.svs')
    parser.add_argument('--thumb_max_dim', type=int, default=2000,
                         help="Max width/height in pixels of the thumbnail used for overlay.")
    parser.add_argument('--alpha', type=float, default=0.5, help="Heatmap opacity over the tissue, 0-1.")
    parser.add_argument('--norm', choices=['rank', 'minmax'], default='rank',
                         help="'rank' (default) spreads values evenly across the colormap regardless of "
                              "skew — fixes an overly blue-dominated look. 'minmax' preserves relative "
                              "attention magnitude but tends to look mostly blue at large L.")
    args = parser.parse_args()

    if args.wsi_dir is not None and not HAS_OPENSLIDE:
        raise RuntimeError("openslide is not installed but --wsi_dir was given. "
                            "pip install openslide-python, or omit --wsi_dir for a bare grid heatmap.")

    with open(args.config, 'r', encoding='utf-8') as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)
    cfg = Munch.fromDict(cfg)
    set_random_seed(cfg.seed)

    device = cfg.device
    save_dir = os.path.join(cfg.save_dir, cfg.config_name, f"fold{args.fold}")

    model = load_model(cfg, save_dir, args.epoch, device)

    with_coords = getattr(cfg.datasets, 'with_coords', False)
    if not with_coords:
        raise RuntimeError("cfg.datasets.with_coords must be True to place heatmaps on the slide.")

    anno_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.wsi_file_path)
    clinical_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.clinical_file_path)
    ids_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.folds_path, f"fold{args.fold}", f"{args.split}.txt")

    ds = DataGeneratorTCGASurvivalWSIEGMLL(
        anno_path, ids_path, clinical_path, shuffle=False, with_coords=with_coords, with_ids=True
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    data, constant_dict = find_slide_batch(loader, args.slide_id)

    per_expert_scores, gate, coords = get_expert_attentions(model, data, device)

    if args.wsi_dir is not None:
        slide_path = os.path.join(args.wsi_dir, args.slide_id + args.slide_ext)
        thumb, scale_x, scale_y = get_thumbnail(slide_path, max_dim=args.thumb_max_dim)
        images = []
        for scores in per_expert_scores:
            scores_norm = normalize(scores, method=args.norm)
            canvas = rasterize_to_thumbnail(
                scores_norm, coords, args.patch_size, thumb.width, thumb.height, scale_x, scale_y
            )
            images.append(overlay_heatmap(thumb, canvas, alpha=args.alpha))
        plot_heatmaps(images, gate, args.slide_id, args.out_dir, is_overlay=True, thumbnail=thumb)
    else:
        canvases = []
        for scores in per_expert_scores:
            scores_norm = normalize(scores, method=args.norm)
            canvas = scores_to_grid(scores_norm, coords, args.patch_size)
            canvases.append(canvas)
        plot_heatmaps(canvases, gate, args.slide_id, args.out_dir, is_overlay=False)


if __name__ == '__main__':
    main()
