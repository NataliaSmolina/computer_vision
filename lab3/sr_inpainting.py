import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import venv
import zipfile
from pathlib import Path

MODEL_ID = "CompVis/ldm-super-resolution-4x-openimages"
DATA_BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
DEGRADATIONS = ("bicubic_x4", "unknown_x4", "realesrgan_like")
MASKS = ("full", "zero", "mixed")
PROJECT_DIR = Path(__file__).resolve().parent / "sr_inpainting_run"
PACKAGES = [
    "numpy>=1.26,<3", "Pillow>=10", "matplotlib>=3.7",
    "torch", "torchvision", "diffusers>=0.34,<0.40", "transformers>=4.44,<5",
    "accelerate>=0.30", "huggingface_hub>=0.30", "safetensors",
    "torchmetrics[image]>=1.4,<2",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local masked diffusion super-resolution lab")
    parser.add_argument("--self-test", action="store_true", help="No dependencies, downloads or model weights")
    parser.add_argument("--no-install", action="store_true", help="Use current Python environment")
    parser.add_argument("--images", type=int, default=3, help="Number of DIV2K validation examples")
    parser.add_argument("--size", type=int, default=256, help="HR crop size, divisible by 128")
    parser.add_argument("--steps", type=int, default=12, help="DDIM denoising steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--out", type=Path, default=PROJECT_DIR)
    parser.add_argument("--hr-dir", type=Path, help="Optional already-downloaded DIV2K HR PNG directory")
    parser.add_argument("--unknown-dir", type=Path, help="Optional DIV2K unknown-X4 PNG directory")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip pretrained LPIPS and CLIP-IQA")
    parser.add_argument("--degradations", nargs="+", choices=DEGRADATIONS, default=list(DEGRADATIONS))
    args = parser.parse_args(argv)
    if not (1 <= args.images <= 100):
        parser.error("--images must be between 1 and 100")
    if args.size < 128 or args.size % 128 != 0:
        parser.error("--size must be >=128 and divisible by 128; 256 recommended")
    if not (1 <= args.steps <= 1000):
        parser.error("--steps must be between 1 and 1000")
    return args


def bootstrap(args):
    """Install everything once into an isolated environment; no global pip edits."""
    if args.self_test or args.no_install or os.environ.get("SR_INPAINTING_IN_VENV") == "1":
        return
    if not (sys.version_info >= (3, 10) and sys.version_info < (3, 14)):
        raise RuntimeError("Use Python 3.10-3.13 (preferably Python 3.11/3.12) for PyTorch on macOS.")
    root = args.out.expanduser().resolve()
    env_dir = root / ".sr_inpainting_venv"
    python_bin = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    root.mkdir(parents=True, exist_ok=True)
    if not python_bin.exists():
        print("[setup] Creating isolated Python environment:", env_dir, flush=True)
        venv.EnvBuilder(with_pip=True).create(env_dir)
    stamp = env_dir / ".packages_installed_v2"
    if not stamp.exists():
        print("[setup] Installing PyTorch, Diffusers, image metrics and plotting packages...", flush=True)
        subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python_bin), "-m", "pip", "install", *PACKAGES], check=True)
        stamp.write_text("Installed requirements:\n" + "\n".join(PACKAGES), encoding="utf-8")
    print("[setup] Launching laboratory inside isolated environment.", flush=True)
    env = os.environ.copy()
    env["SR_INPAINTING_IN_VENV"] = "1"
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Preserve all CLI options, including caller-defined output paths.
    outcome = subprocess.run([str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    sys.exit(outcome.returncode)


def dep_imports():
    global np, Image, ImageFilter, torch, F, plt
    import numpy as np
    from PIL import Image, ImageFilter
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def bicubic():
    return Image.Resampling.BICUBIC


def build_mask(kind, size):
    mask = np.zeros((size, size), dtype=np.uint8)
    if kind == "full":
        mask[:] = 1
    elif kind == "mixed":
        mask[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 1
    elif kind != "zero":
        raise ValueError("Unknown mask type: " + str(kind))
    return mask


def composite(source, generated, mask):
    """Exact uint8 pixel copy outside the binary mask."""
    if source.shape != generated.shape or source.shape[:2] != mask.shape:
        raise ValueError("Mismatched source/generated/mask shapes")
    if source.dtype != np.uint8 or generated.dtype != np.uint8:
        raise TypeError("uint8 RGB images required")
    result = source.copy()
    result[mask.astype(bool)] = generated[mask.astype(bool)]
    return result


def psnr(gt, prediction, mask=None):
    diff = (gt.astype(np.float64) - prediction.astype(np.float64)) ** 2
    if mask is not None:
        if mask.shape != gt.shape[:2]:
            raise ValueError("PSNR region shape mismatch")
        region = mask.astype(bool)
        if not region.any():
            return float("nan")
        diff = diff[region]
    mse = float(np.mean(diff))
    return math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)


def to_tensor(array, device):
    arr = np.asarray(array)
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32
    ) / 127.5 - 1


def to_rgb(tensor):
    return ((tensor[0].detach().float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1)
            * 127.5).round().clip(0, 255).astype(np.uint8)


def save_rgb(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), "RGB").save(path)


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(path)


def zip_valid(path):
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def download_zip(urls, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and zip_valid(path):
        print("[dataset] Cached:", path.name, flush=True)
        return path
    path.unlink(missing_ok=True)
    errors = []
    for url in urls:
        for attempt in range(1, 3):
            part = path.with_suffix(".zip.part")
            try:
                print("[dataset] Downloading:", url, "attempt", attempt, flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as response, open(part, "wb") as dest:
                    total = int(response.headers.get("Content-Length") or 0)
                    written = 0
                    next_report = 50 * 1024 * 1024
                    while True:
                        buf = response.read(1024 * 1024)
                        if not buf:
                            break
                        dest.write(buf)
                        written += len(buf)
                        if written >= next_report:
                            print("[dataset] {:.1f} MB{}".format(
                                written / 1e6, " / {:.1f} MB".format(total / 1e6) if total else ""), flush=True)
                            next_report += 50 * 1024 * 1024
                if not zip_valid(part):
                    raise ValueError("Archive failed ZIP integrity check")
                part.replace(path)
                return path
            except Exception as exc:
                errors.append(repr(exc))
                print("[dataset] Download attempt failed:", repr(exc), flush=True)
                part.unlink(missing_ok=True)
    raise RuntimeError("Could not download {}: {}".format(path.name, "; ".join(errors)))


def find_png(folder, stem):
    if folder is None or not folder.exists():
        return None
    for path in folder.rglob("*.png"):
        if path.stem.lower() == stem.lower():
            return path
    return None


def extract_selected(archive, target, wanted):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        entry_map = {Path(name).name.lower(): name for name in zf.namelist() if name.lower().endswith(".png")}
        for filename in wanted:
            out = target / filename
            if out.exists():
                continue
            member = entry_map.get(filename.lower())
            if member is None:
                raise FileNotFoundError("Missing {} in {}".format(filename, archive))
            with zf.open(member) as src, open(out, "wb") as dest:
                shutil.copyfileobj(src, dest)


def div2k_examples(args, data_dir):
    hr_dir = args.hr_dir.expanduser().resolve() if args.hr_dir else data_dir / "valid_hr"
    unknown_dir = args.unknown_dir.expanduser().resolve() if args.unknown_dir else data_dir / "valid_unknown_x4"
    ids = [f"{i:04d}" for i in range(801, 801 + args.images)]
    missing_hr = [f"{i}.png" for i in ids if find_png(hr_dir, i) is None]
    if missing_hr:
        if args.hr_dir:
            raise FileNotFoundError("Missing HR files in {}: {}".format(hr_dir, missing_hr))
        archive = download_zip([DATA_BASE + "DIV2K_valid_HR.zip",
                                "https://huggingface.co/datasets/deepinv/DIV2K/resolve/main/DIV2K_valid_HR.zip"],
                               data_dir / "archives" / "DIV2K_valid_HR.zip")
        extract_selected(archive, hr_dir, missing_hr)
        archive.unlink(missing_ok=True)  # Save limited Mac SSD space after extraction.
    if "unknown_x4" in args.degradations:
        missing_lr = [f"{i}x4.png" for i in ids if find_png(unknown_dir, i + "x4") is None]
        if missing_lr:
            if args.unknown_dir:
                raise FileNotFoundError("Missing unknown-X4 files in {}: {}".format(unknown_dir, missing_lr))
            archive = download_zip([DATA_BASE + "DIV2K_valid_LR_unknown_X4.zip"],
                                   data_dir / "archives" / "DIV2K_valid_LR_unknown_X4.zip")
            extract_selected(archive, unknown_dir, missing_lr)
            archive.unlink(missing_ok=True)
    return [(i, find_png(hr_dir, i), find_png(unknown_dir, i + "x4")) for i in ids]


def crop_pair(hr, unknown, size):
    if min(hr.size) < size:
        raise ValueError("DIV2K image smaller than selected crop")
    left = (((hr.width - size) // 2) // 4) * 4
    top = (((hr.height - size) // 2) // 4) * 4
    hr = hr.crop((left, top, left + size, top + size))
    if unknown is not None:
        if unknown.width < (left + size) // 4 or unknown.height < (top + size) // 4:
            raise ValueError("Unknown-X4 LR size does not align with HR image")
        unknown = unknown.crop((left // 4, top // 4, (left + size) // 4, (top + size) // 4))
    return hr, unknown


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade(hr, kind, rng, unknown=None):
    """Return LR=H/4 by W/4 and corresponding bicubic-upsampled input."""
    size = hr.width
    lr_shape = (size // 4, size // 4)
    if kind == "bicubic_x4":
        lr = hr.resize(lr_shape, bicubic())
    elif kind == "unknown_x4":
        if unknown is None:
            raise ValueError("DIV2K unknown-X4 pair not available")
        if unknown.size != lr_shape:
            raise ValueError("DIV2K unknown-X4 crop size mismatch")
        lr = unknown.copy()
    elif kind == "realesrgan_like":
        # Simplified synthetic Real-ESRGAN-style; NOT the official Real-ESRGAN degradation.
        x = hr.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 2.2))))
        mid = int(rng.integers(max(size // 3, 64), size + 1))
        x = x.resize((mid, mid), bicubic())
        arr = np.asarray(x, dtype=np.float32).copy()
        arr += rng.normal(0, rng.uniform(1.0, 8.0), arr.shape)
        x = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")
        x = jpeg(x, rng.integers(35, 90))
        x = x.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.1, 0.8))))
        lr = jpeg(x.resize(lr_shape, bicubic()), rng.integers(40, 95))
    else:
        raise ValueError("Unknown degradation " + str(kind))
    lr_array = np.asarray(lr.convert("RGB"), dtype=np.uint8).copy()
    input_array = np.asarray(lr.resize(hr.size, bicubic()), dtype=np.uint8).copy()
    return lr_array, input_array


def choose_device(kind):
    if kind == "auto":
        if torch.backends.mps.is_available():
            kind = "mps"
        elif torch.cuda.is_available():
            kind = "cuda"
        else:
            kind = "cpu"
    if kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS unavailable. Use macOS on Apple Silicon with a supported PyTorch build.")
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return torch.device(kind)


def load_pipeline(device):
    from diffusers import DDIMScheduler, LDMSuperResolutionPipeline
    print("[model] Downloading/loading", MODEL_ID, flush=True)
    # fp32 is safer across Apple MPS backends. The model is ~0.7 GB of weights.
    pipe = LDMSuperResolutionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.eval()
    pipe.vqvae.eval()
    pipe = pipe.to(device)
    if int(pipe.unet.config.in_channels) != 6:
        raise RuntimeError("Expected 6 input channels (3 latent + 3 LR condition)")
    if len(pipe.vqvae.config.block_out_channels) != 3:
        raise RuntimeError("Unexpected VQ-VAE 4x upscaling architecture")
    print("[model] Loaded: device={} dtype=float32, DDIM scheduler".format(device), flush=True)
    return pipe


def seed_noise(shape, seed, device):
    # CPU generator makes noise seed portable to MPS, which lacks some generator ops.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(device)


def denoise_with_mask(pipe, lowres, source, mask, steps, seed, device, trace=None):
    """Modified version of the upstream LDMSuperResolutionPipeline sampling.

    Input condition: genuine LR pixels, normalized to [-1, 1].
    Diffusion latent: LR spatial size, 3 channels; VQ decoder upscales x4.

    Mask: HxW binary pixels in the OUTPUT (HR) image space.
    For each DDIM transition, anchor unmasked latent to the forward-noised
    VQ-encoding of the *bicubic-upscaled INPUT* (NEVER HR ground truth).
    Final RGB pixel composition guarantees exact preservation outside ROI.
    """
    if lowres.shape[0] * 4 != source.shape[0] or lowres.shape[1] * 4 != source.shape[1]:
        raise ValueError("Expected exact x4 LR/HR relation")
    if mask.shape != source.shape[:2] or not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask must be a binary HxW image-size array")
    if not mask.any():
        return source.copy()
    h, w = lowres.shape[:2]
    condition = to_tensor(lowres, device)
    mask_latent = F.interpolate(torch.as_tensor(mask, device=device, dtype=torch.float32)[None, None],
                                size=(h, w), mode="nearest")
    mask_latent = mask_latent.to(torch.float32)
    src_reference = to_tensor(source, device) if not mask.all() else None
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=device)
    initial = seed_noise((1, 3, h, w), seed, device) * scheduler.init_noise_sigma
    latents = initial
    reference_latent = None
    reference_noise = None
    if src_reference is not None:
        with torch.inference_mode():
            reference_latent = pipe.vqvae.encode(src_reference).latents.detach()
        if tuple(reference_latent.shape) != tuple(latents.shape):
            raise RuntimeError("Reference VQ latent shape mismatch: {} vs {}".format(
                reference_latent.shape, latents.shape))
        reference_noise = seed_noise(tuple(latents.shape), seed + 104729, device)
        # Align the *initial* outside region with the DDIM forward process.
        t_start = scheduler.timesteps[0]
        noisy_reference = scheduler.add_noise(reference_latent, reference_noise,
                                              t_start.reshape(1))
        latents = mask_latent * latents + (1 - mask_latent) * noisy_reference
    with torch.inference_mode():
        for index, timestep in enumerate(scheduler.timesteps):
            denoise_input = torch.cat([latents, condition], dim=1)
            denoise_input = scheduler.scale_model_input(denoise_input, timestep)
            predicted = pipe.unet(denoise_input, timestep).sample
            latents = scheduler.step(predicted, timestep, latents, eta=0.0).prev_sample
            # Critical lab modification: projection performed INSIDE EACH step.
            if reference_latent is not None:
                if index + 1 < len(scheduler.timesteps):
                    next_t = scheduler.timesteps[index + 1]
                    reference_t = scheduler.add_noise(reference_latent, reference_noise,
                                                      next_t.reshape(1))
                else:
                    reference_t = reference_latent
                latents = mask_latent * latents + (1 - mask_latent) * reference_t
                if trace is not None:
                    outside_error = (latents * (1 - mask_latent) -
                                     reference_t * (1 - mask_latent)).abs().max().item()
                    trace.append(float(outside_error))
        generated = pipe.vqvae.decode(latents).sample
        generated = to_rgb(generated)
    if generated.shape != source.shape:
        raise RuntimeError("VQ decoder returned {} but expected {}".format(generated.shape, source.shape))
    return composite(source, generated, mask)


def plot_case(gt, source, baseline, post, masked, mask, title, filename):
    panels = [("HR reference", gt), ("Degraded input", source),
              ("Ordinary LDM SR", baseline), ("Mask (white = changed)", mask * 255),
              ("Output-only blending", post), ("Masked DDIM sampling", masked)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (label, image) in zip(axes.flat, panels):
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None,
                  vmin=0 if image.ndim == 2 else None, vmax=255 if image.ndim == 2 else None)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def roi_bbox(image, mask):
    yy, xx = np.where(mask.astype(bool))
    if not yy.size:
        return None
    return image[int(yy.min()):int(yy.max())+1, int(xx.min()):int(xx.max())+1].copy()


class ImageMetrics:
    """Pretrained LPIPS and CLIP-IQA on CPU to leave MPS memory for diffusion."""

    def __init__(self):
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.multimodal import CLIPImageQualityAssessment
        print("[metrics] Loading pretrained LPIPS (AlexNet) on CPU...", flush=True)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cpu().eval()
        print("[metrics] Loading CLIP-IQA (CLIP ViT-B/16, quality prompts) on CPU...", flush=True)
        self.clip = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-base-patch16",
                                               prompts=("quality",)).cpu().eval()
        self.clip_cache = {}
        self.lpips_cache = {}

    @staticmethod
    def as_tensor(image):
        # Metric model handles its own CLIP preprocessing; 0..1 range.
        arr = np.ascontiguousarray(image)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    @staticmethod
    def digest(image):
        return hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()

    def clip_iqa(self, image):
        key = self.digest(image)
        if key not in self.clip_cache:
            with torch.inference_mode():
                value = self.clip(self.as_tensor(image))
                self.clip_cache[key] = float(value.reshape(-1)[0].item())
                self.clip.reset()
        return self.clip_cache[key]

    def lpips_value(self, gt, candidate):
        key = (self.digest(gt), self.digest(candidate))
        if key not in self.lpips_cache:
            with torch.inference_mode():
                value = self.lpips(self.as_tensor(gt), self.as_tensor(candidate))
                self.lpips_cache[key] = float(value.item())
                self.lpips.reset()
        return self.lpips_cache[key]


def metrics_for(gt, prediction, source, mask, metric_models):
    outside = mask == 0
    outside_max = int(np.abs(prediction[outside].astype(np.int16) -
                             source[outside].astype(np.int16)).max()) if outside.any() else 0
    values = {
        "psnr": psnr(gt, prediction), "psnr_roi": psnr(gt, prediction, mask),
        "max_outside_error": outside_max,
        "lpips": float("nan"), "clip_iqa": float("nan"),
        "lpips_roi_bbox": float("nan"), "clip_iqa_roi_bbox": float("nan"),
    }
    if metric_models is not None:
        values["lpips"] = metric_models.lpips_value(gt, prediction)
        values["clip_iqa"] = metric_models.clip_iqa(prediction)
        a = roi_bbox(gt, mask)
        b = roi_bbox(prediction, mask)
        if a is not None and min(a.shape[:2]) >= 64:
            values["lpips_roi_bbox"] = metric_models.lpips_value(a, b)
            values["clip_iqa_roi_bbox"] = metric_models.clip_iqa(b)
    return values


def avg_finite(numbers):
    finite = [float(v) for v in numbers if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def create_summary(rows, plot_dir, result_dir):
    aggregate = []
    for method in ("post_only", "masked_sampling"):
        for degradation in DEGRADATIONS:
            for mask in MASKS:
                selected = [r for r in rows if r["method"] == method and
                            r["degradation"] == degradation and r["mask"] == mask]
                if not selected:
                    continue
                row = {"method": method, "degradation": degradation, "mask": mask,
                       "n_images": len(selected)}
                for key in ("psnr", "psnr_roi", "lpips", "clip_iqa",
                            "lpips_roi_bbox", "clip_iqa_roi_bbox"):
                    row[key] = avg_finite([r[key] for r in selected])
                aggregate.append(row)
    write_csv(result_dir / "summary.csv", aggregate)
    for metric, title in (("psnr", "PSNR ↑"), ("psnr_roi", "ROI PSNR ↑"),
                          ("lpips", "LPIPS ↓"), ("clip_iqa", "CLIP-IQA ↑"),
                          ("lpips_roi_bbox", "ROI bounding-box LPIPS ↓"),
                          ("clip_iqa_roi_bbox", "ROI bounding-box CLIP-IQA ↑")):
        valid = [x for x in aggregate if x["method"] == "masked_sampling"]
        labels = list(dict.fromkeys(r["degradation"] for r in valid))
        if not labels or not any(math.isfinite(float(x[metric])) for x in valid):
            continue
        x_positions = np.arange(len(labels), dtype=float)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for index, mask in enumerate(MASKS):
            heights = [next((r[metric] for r in valid if r["mask"] == mask
                             and r["degradation"] == d), float("nan")) for d in labels]
            ax.bar(x_positions + (index - 1) * .24, heights, width=.24, label=mask)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(title)
        ax.set_title(title + " — masked DDIM")
        ax.grid(axis="y", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / (metric + "_by_degradation.png"), dpi=145)
        plt.close(fig)
    for metric, title in (("psnr_roi", "ROI PSNR ↑"),
                          ("lpips_roi_bbox", "ROI LPIPS ↓"),
                          ("clip_iqa_roi_bbox", "ROI CLIP-IQA ↑")):
        options = [r for r in aggregate if r["mask"] == "mixed"]
        labels = list(dict.fromkeys(r["degradation"] for r in options))
        if not labels or not any(math.isfinite(float(r[metric])) for r in options):
            continue
        x_positions = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for index, method in enumerate(("post_only", "masked_sampling")):
            heights = [next((r[metric] for r in options if r["method"] == method
                             and r["degradation"] == d), float("nan")) for d in labels]
            ax.bar(x_positions + (index - .5) * .35, heights, width=.35, label=method)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_title("Ablation: " + title)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / ("ablation_" + metric + ".png"), dpi=145)
        plt.close(fig)
    return aggregate


def emit_readme(args, root):
    text = f"""# SR-inpainting — local LDM experiment

## Experimental setup

- Pretrained model: `{MODEL_ID}` (NOT SUPIR; SUPIR is optional in the assignment).
- Local device: Apple MPS when available; otherwise CUDA or CPU.
- Inference: DDIM, {args.steps} steps, RNG seed {args.seed}.
- Dataset: first {args.images} examples from DIV2K validation set (0801 onwards).
- HR crop: {args.size}×{args.size}; matching LR crop: {args.size // 4}×{args.size // 4}.
- Degradations: {', '.join(args.degradations)}.
- Masks: full (all 1), zero (all 0), mixed (center 25% of HR image).

## What was changed in the sampler

Original LDM SR has a DDIM loop: `z[t-1] = scheduler.step(unet([z[t], LR], t))`.
For a binary HR mask M, downsample M with nearest-neighbor interpolation to latent
resolution. Encode the *bicubic-upsampled degraded input* through VQ-VAE as
`z_ref`, and sample fixed `eps_ref`. After EACH DDIM transition, replace outside
mask with the corresponding forward-noised reference:

`z[t-1] = M * z_generated[t-1] + (1-M) * (sqrt(alpha[t-1])*z_ref + sqrt(1-alpha[t-1])*eps_ref)`

For the last timestep outside the mask use `z_ref` directly. Output RGB is then
composed pixelwise with the degraded input, which guarantees exact equality
outside ROI. Ground-truth HR is NEVER used as a sampling reference.

Full mask uses unmodified DDIM (ordinary SR). Zero mask skips inference and
returns the degraded input. Mixed mask uses changed sampling, not just blending.
For ablation, `post_only` applies the mask after ordinary, unmodified DDIM.

## Metrics

- PSNR full / ROI (ROI PSNR undefined for an empty mask).
- LPIPS (AlexNet) full; ROI *bounding-box* LPIPS (not pixel-wise masked LPIPS).
- CLIP-IQA (ViT-B/16 quality prompt) full; ROI *bounding-box* CLIP-IQA.
- max_outside_error must always equal zero for all masked-sampling outputs.
- PSNR ↑ is better; LPIPS ↓ better; CLIP-IQA ↑ is better.

## Important limits

- `realesrgan_like` is only a simplified Real-ESRGAN-style degradation: blur,
  resampling, noise and JPEG. It is NOT the exact official Real-ESRGAN pipeline.
- Output size is x4 compared to the LR input. This example is a controlled
  256-pixel-ish DIV2K crop, not full-resolution restoration of original photos.
- Low diffusion steps trade output quality for practicality on a MacBook Air.
- CLIP-IQA is a *no-reference* perceptual score; it does not measure GT fidelity.
- A model or library download failure is reported explicitly, not replaced by
  synthetic outputs or fabricated scores.

## Outputs

- `results/<image_id>/<degradation>/` contains HR, LR, degraded, mask, baseline,
  post-only and masked results.
- `results/metrics.csv` per-case scores; `results/summary.csv` aggregate.
- `plots/compare_*.png` visual comparisons; `plots/*_by_degradation.png`
  and `plots/ablation_*.png` metric plots (when metrics available).

## Sources

- https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py
- https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def self_test():
    """Check shape/ROI algorithms, and mock sampler if PyTorch is already installed."""
    dep_imports()
    size = 256
    a = np.zeros((size, size, 3), dtype=np.uint8)
    b = np.full_like(a, 217)
    assert math.isinf(psnr(a, a))
    for name in MASKS:
        m = build_mask(name, size)
        assert m.shape == (size, size)
        assert set(np.unique(m).tolist()).issubset({0, 1})
        out = composite(a, b, m)
        assert np.array_equal(out[m == 0], a[m == 0])
        assert np.array_equal(out[m == 1], b[m == 1])
        if name == "zero":
            assert np.array_equal(out, a)
            assert math.isnan(psnr(a, b, m))
    from PIL import Image
    test_image = Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8))
    for kind in ("bicubic_x4", "realesrgan_like"):
        lr, up = degrade(test_image, kind, np.random.default_rng(9))
        assert lr.shape == (64, 64, 3) and up.shape == (256, 256, 3)
    # Validate DDIM mask-projection branch with cheap fake components.
    from types import SimpleNamespace
    class FakeScheduler:
        init_noise_sigma = 1.0
        timesteps = None
        def set_timesteps(self, steps, device=None):
            self.timesteps = torch.arange(steps - 1, -1, -1, device=device)
        def add_noise(self, latents, noise, timesteps):
            return latents + noise * (timesteps[0] + 1) / 100
        def scale_model_input(self, latents, timestep):
            return latents
        def step(self, prediction, timestep, latents, eta=0.0):
            return SimpleNamespace(prev_sample=latents - prediction * 0.1)
    class FakeUnet:
        def __call__(self, sample, timestep):
            return SimpleNamespace(sample=sample[:, :3] * .1)
    class FakeVQ:
        def encode(self, sample):
            return SimpleNamespace(latents=F.avg_pool2d(sample, 4))
        def decode(self, sample):
            return SimpleNamespace(sample=F.interpolate(sample, scale_factor=4,
                                                       mode="nearest"))
    fake = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUnet(), vqvae=FakeVQ())
    lr = np.full((64, 64, 3), 128, dtype=np.uint8)
    source = np.full((256, 256, 3), 128, dtype=np.uint8)
    m = build_mask("mixed", 256)
    trace = []
    out = denoise_with_mask(fake, lr, source, m, 4, 42, torch.device("cpu"), trace)
    assert len(trace) == 4 and all(x < 1e-6 for x in trace), trace
    assert np.array_equal(out[m == 0], source[m == 0])
    assert np.array_equal(denoise_with_mask(fake, lr, source, build_mask("zero", 256),
                                           4, 42, torch.device("cpu")), source)
    print("PASS: 3 binary masks; x4 degradation; outside-pixel invariance; PSNR;",
          "DDIM mock with inside-loop reference anchoring")
    print("NOTE: this is a synthetic unit test, NOT a full pretrained model run.")


def run(args):
    dep_imports()
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    root = args.out.expanduser().resolve()
    data = root / "data"
    output = root / "results"
    plots = root / "plots"
    for folder in (data, output, plots):
        folder.mkdir(parents=True, exist_ok=True)
    emit_readme(args, root)
    print("[system] {}, Python {}, PyTorch {}".format(platform.platform(),
                                                     platform.python_version(), torch.__version__), flush=True)
    device = choose_device(args.device)
    print("[system] Device:", device, "— MPS uses Apple GPU without CUDA", flush=True)
    examples = div2k_examples(args, data)
    pipe = load_pipeline(device)
    rows = []
    print("[run] Images={}, degradations={}, masks={}, steps={} (cached files reused)".format(
        len(examples), ",".join(args.degradations), ",".join(MASKS), args.steps), flush=True)
    for image_pos, (image_id, hr_path, unknown_path) in enumerate(examples):
        with Image.open(hr_path) as f:
            full_hr = f.convert("RGB")
        unknown = None
        if unknown_path is not None and unknown_path.exists():
            with Image.open(unknown_path) as f:
                unknown = f.convert("RGB")
        hr, unknown_crop = crop_pair(full_hr, unknown, args.size)
        gt = np.asarray(hr, dtype=np.uint8).copy()
        for mode_pos, mode in enumerate(args.degradations):
            rng = np.random.default_rng(args.seed + image_pos * 1000 + mode_pos * 100)
            lowres, source = degrade(hr, mode, rng, unknown_crop)
            folder = output / image_id / mode
            folder.mkdir(parents=True, exist_ok=True)
            # Avoid reusing outputs made with different steps, seed, crops or inputs.
            signature = {
                "model": MODEL_ID, "size": args.size, "steps": args.steps,
                "seed": args.seed, "degradation": mode,
                "lr_sha256": hashlib.sha256(lowres.tobytes()).hexdigest(),
                "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            }
            manifest_file = folder / "case_manifest.json"
            if manifest_file.exists():
                try:
                    previous = json.loads(manifest_file.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    previous = None
            else:
                previous = None
            if previous != signature:
                for old in [folder / "ordinary_ldm_sr.png", *folder.glob("masked_*.png")]:
                    old.unlink(missing_ok=True)
                manifest_file.write_text(json.dumps(signature, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
            save_rgb(folder / "hr.png", gt)
            save_rgb(folder / "lr.png", lowres)
            save_rgb(folder / "degraded_input.png", source)
            baseline_path = folder / "ordinary_ldm_sr.png"
            if baseline_path.exists():
                baseline = np.asarray(Image.open(baseline_path).convert("RGB"), dtype=np.uint8).copy()
                print("[resume] Ordinary SR:", folder, flush=True)
            else:
                print("[sampling] image={}, degradation={}: ordinary diffusion".format(image_id, mode), flush=True)
                baseline = denoise_with_mask(pipe, lowres, source, build_mask("full", args.size),
                                            args.steps, args.seed, device)
                save_rgb(baseline_path, baseline)
            for kind in MASKS:
                mask = build_mask(kind, args.size)
                save_mask(folder / ("mask_" + kind + ".png"), mask)
                post = composite(source, baseline, mask)
                masked_path = folder / ("masked_" + kind + ".png")
                if kind == "zero":
                    masked = source.copy()
                elif kind == "full":
                    masked = baseline.copy()
                elif masked_path.exists():
                    masked = np.asarray(Image.open(masked_path).convert("RGB"), dtype=np.uint8).copy()
                    print("[resume] Masked SR:", folder, flush=True)
                else:
                    print("[sampling] image={}, degradation={}: modified DDIM / mixed mask".format(
                        image_id, mode), flush=True)
                    masked = denoise_with_mask(pipe, lowres, source, mask, args.steps,
                                              args.seed, device)
                save_rgb(folder / ("post_only_" + kind + ".png"), post)
                save_rgb(masked_path, masked)
                assert np.array_equal(masked[mask == 0], source[mask == 0])
                if kind == "zero":
                    assert np.array_equal(masked, source)
                elif kind == "full":
                    assert np.array_equal(masked, baseline)
                plot_case(gt, source, baseline, post, masked, mask,
                          f"DIV2K {image_id} | {mode} | mask {kind}",
                          plots / f"compare_{image_id}_{mode}_{kind}.png")
            print("[saved] image={}, degradation={} (3 masks + plots)".format(image_id, mode), flush=True)
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
    # Free diffusion model before loading pretrained quality-evaluation models.
    del pipe
    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    scorer = None if args.skip_metrics else ImageMetrics()
    for image_id, hr_path, unknown_path in examples:
        for mode in args.degradations:
            folder = output / image_id / mode
            gt = np.asarray(Image.open(folder / "hr.png").convert("RGB"), dtype=np.uint8)
            source = np.asarray(Image.open(folder / "degraded_input.png").convert("RGB"), dtype=np.uint8)
            for kind in MASKS:
                mask = build_mask(kind, args.size)
                for method in ("post_only", "masked_sampling"):
                    candidate_name = ("post_only_" if method == "post_only" else "masked_") + kind + ".png"
                    candidate = np.asarray(Image.open(folder / candidate_name).convert("RGB"), dtype=np.uint8)
                    metric_values = metrics_for(gt, candidate, source, mask, scorer)
                    rows.append({"image_id": image_id, "degradation": mode, "mask": kind,
                                 "method": method, "seed": args.seed, "steps": args.steps,
                                 "mask_fraction": float(mask.mean()), **metric_values})
                write_csv(output / "metrics.csv", rows)
            print("[metrics] evaluated:", image_id, mode, flush=True)
    create_summary(rows, plots, output)
    print("\n[DONE] All requested cases processed.", flush=True)
    print("[DONE] Result images:", output, flush=True)
    print("[DONE] Comparisons and graphs:", plots, flush=True)
    print("[DONE] CSV:", output / "metrics.csv", flush=True)
    print("[DONE] Experiment description:", root / "README.md", flush=True)


if __name__ == "__main__":
    params = parse_args()
    if params.self_test:
        dep_imports()
        self_test()
    else:
        bootstrap(params)
        run(params)
