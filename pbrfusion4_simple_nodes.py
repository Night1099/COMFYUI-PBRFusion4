"""
AIXPOLY PBRFusion4 Simple Depth & Normal Map Generator Nodes
Simplified version without seamless processing
"""

import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageFilter
from contextlib import nullcontext
import folder_paths
import comfy.model_management as mm
import comfy.utils

# Add Lotus directory to path
LOTUS_DIR = Path(__file__).parent / "Lotus"
if str(LOTUS_DIR) not in sys.path:
    sys.path.insert(0, str(LOTUS_DIR))

from pipeline import LotusDPipeline

# Register custom model folder for PBRFusion4
PBRFUSION4_MODEL_DIR = os.path.join(folder_paths.models_dir, "pbrfusion4")
os.makedirs(PBRFUSION4_MODEL_DIR, exist_ok=True)

# Model paths - single safetensors (preferred) or diffusers folder (fallback)
PBRFUSION4_SAFETENSORS_PATH = os.path.join(PBRFUSION4_MODEL_DIR, "PBRFusion4.safetensors")
PBRFUSION4_MODEL_PATH = os.path.join(PBRFUSION4_MODEL_DIR, "LotusDInstNorm")
PBRFUSION4_DOWNLOAD_URL = "https://huggingface.co/NightRaven109/PBRFusion4/resolve/main/PBRFusion4.safetensors"

# Global model cache
_PBRFUSION4_MODELS = {}


def get_device():
    """Get the appropriate device for model execution"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _load_from_single_safetensors(safetensors_path, dtype, device):
    """
    Load LotusDPipeline from a single .safetensors file.

    The file contains all weights (unet/vae/text_encoder) with prefix keys,
    and all configs + tokenizer files embedded as metadata.
    """
    from safetensors.torch import load_file, safe_open
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

    print(f"Loading pbrfusion4: {safetensors_path}")

    # Read metadata (configs + tokenizer files)
    metadata = {}
    with safe_open(safetensors_path, framework="pt") as f:
        metadata = f.metadata()

    # Load all weights
    all_weights = load_file(safetensors_path, device="cpu")

    # Split weights by component prefix
    unet_sd = {k[len("unet."):]: v for k, v in all_weights.items() if k.startswith("unet.")}
    vae_sd = {k[len("vae."):]: v for k, v in all_weights.items() if k.startswith("vae.")}
    te_sd = {k[len("text_encoder."):]: v for k, v in all_weights.items() if k.startswith("text_encoder.")}
    del all_weights

    # Reconstruct UNet from config
    unet_config = json.loads(metadata["unet/config.json"])
    unet = UNet2DConditionModel(**{k: v for k, v in unet_config.items() if not k.startswith("_")})
    unet.load_state_dict(unet_sd)
    del unet_sd

    # Reconstruct VAE from config
    vae_config = json.loads(metadata["vae/config.json"])
    vae = AutoencoderKL(**{k: v for k, v in vae_config.items() if not k.startswith("_")})
    vae.load_state_dict(vae_sd)
    del vae_sd

    # Reconstruct text encoder from config
    from transformers import CLIPTextConfig
    te_config_dict = json.loads(metadata["text_encoder/config.json"])
    te_config = CLIPTextConfig(**te_config_dict)
    text_encoder = CLIPTextModel(te_config)
    text_encoder.load_state_dict(te_sd)
    del te_sd

    # Reconstruct tokenizer: write temp files from metadata
    tokenizer_files = {k: v for k, v in metadata.items() if k.startswith("tokenizer/")}
    tmpdir = tempfile.mkdtemp(prefix="lotus_tokenizer_")
    for key, content in tokenizer_files.items():
        filename = key.split("/", 1)[1]
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    tokenizer = CLIPTokenizer.from_pretrained(tmpdir)

    # Reconstruct scheduler from config (use from_config to ignore unknown keys)
    sched_key = "scheduler/scheduler_config.json"
    if sched_key not in metadata:
        sched_key = "scheduler/config.json"
    sched_config = json.loads(metadata[sched_key])
    scheduler = DDIMScheduler.from_config(sched_config)

    # Feature extractor (optional, used by safety checker which is disabled)
    feature_extractor = CLIPImageProcessor()

    # Assemble pipeline
    pipe = LotusDPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=feature_extractor,
        requires_safety_checker=False,
    )

    pipe.to(dtype=dtype, device=device)
    return pipe


def _download_model(url, dest_path):
    """Download model file with progress reporting."""
    import urllib.request

    print(f"Downloading PBRFusion4 model...")
    print(f"  From: {url}")
    print(f"  To:   {dest_path}")

    tmp_path = dest_path + ".tmp"
    last_percent = [0]

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = int(block_num * block_size * 100 / total_size)
            if percent >= last_percent[0] + 10:
                last_percent[0] = percent
                mb_done = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"  {percent}% ({mb_done:.0f}/{mb_total:.0f} MB)")

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=progress_hook)
        os.replace(tmp_path, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Download complete ({size_mb:.0f} MB)")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_pbrfusion4_models(task_name='depth', device=None):
    """
    Load PBRFusion4 model pipeline.

    Tries in order:
      1. Single safetensors: ComfyUI/models/pbrfusion4/PBRFusion4.safetensors
      2. Diffusers folder:   ComfyUI/models/pbrfusion4/LotusDInstNorm/
      3. Auto-download safetensors from HuggingFace

    Args:
        task_name: 'depth' or 'normal'
        device: torch device, auto-detected if None

    Returns:
        tuple: (None, pipe_d) - only discriminative pipeline is used
    """
    if device is None:
        device = mm.get_torch_device()

    cache_key = f"{task_name}_{device}"

    if cache_key in _PBRFUSION4_MODELS:
        return _PBRFUSION4_MODELS[cache_key]

    dtype = torch.float16

    # Auto-download if neither format exists
    if not os.path.exists(PBRFUSION4_SAFETENSORS_PATH) and not os.path.exists(PBRFUSION4_MODEL_PATH):
        _download_model(PBRFUSION4_DOWNLOAD_URL, PBRFUSION4_SAFETENSORS_PATH)

    if os.path.exists(PBRFUSION4_SAFETENSORS_PATH):
        pipe_d = _load_from_single_safetensors(PBRFUSION4_SAFETENSORS_PATH, dtype, device)
    elif os.path.exists(PBRFUSION4_MODEL_PATH):
        print(f"Loading pbrfusion4 from diffusers dir: {PBRFUSION4_MODEL_PATH}")
        pipe_d = LotusDPipeline.from_pretrained(
            PBRFUSION4_MODEL_PATH,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe_d.to(device)
    else:
        raise FileNotFoundError(
            f"Model not found at {PBRFUSION4_SAFETENSORS_PATH} or {PBRFUSION4_MODEL_PATH}"
        )

    pipe_d.set_progress_bar_config(disable=True)
    pipe_g = None

    _PBRFUSION4_MODELS[cache_key] = (pipe_g, pipe_d)
    print(f"PBRFusion4 {task_name} model loaded successfully")

    return pipe_g, pipe_d


def apply_gaussian_blur(image, radius=1.0):
    """Apply Gaussian blur to PIL Image with specified radius"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def get_image_intensity(img, gamma_correction=1.0):
    """
    Extract intensity map from an image using HSV color space

    Args:
        img: numpy array (RGB)
        gamma_correction: gamma correction factor

    Returns:
        numpy array: intensity map in RGB format
    """
    # Convert to HSV color space
    result = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Extract Value channel (intensity)
    result = result[:, :, 2].astype(np.float32) / 255.0
    # Apply gamma correction
    result = result ** gamma_correction
    # Convert back to 0-255 range
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    # Convert to RGB (still grayscale but in RGB format)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def blend_numpy_images(image1, image2, blend_factor=0.25):
    """
    Blend two numpy images using normal mode

    Args:
        image1: First image (numpy array)
        image2: Second image (numpy array)
        blend_factor: Blend factor (0-1)

    Returns:
        numpy array: Blended image
    """
    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        # Resize both to the higher dimension
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        target_height = max(h1, h2)
        target_width = max(w1, w2)

        if (h1, w1) != (target_height, target_width):
            image1 = cv2.resize(image1, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        if (h2, w2) != (target_height, target_width):
            image2 = cv2.resize(image2, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 and normalize to 0-1
    img1 = image1.astype(np.float32) / 255.0
    img2 = image2.astype(np.float32) / 255.0

    # Normal blend mode
    blended = img1 * (1 - blend_factor) + img2 * blend_factor

    # Convert back to uint8
    blended = (blended * 255.0).clip(0, 255).astype(np.uint8)
    return blended


def compute_normal_from_height(height_map, scale=2.0):
    """
    Compute normal map from height map using Sobel operators (matching reference implementation)

    Args:
        height_map: torch.Tensor, single-channel height map (H, W) in range [0, 1]
        scale: float, scaling factor for gradients

    Returns:
        torch.Tensor: normal map (3, H, W) in range [-1, 1]
    """
    if height_map.dim() == 2:
        height_map = height_map.unsqueeze(0).unsqueeze(0)
    elif height_map.dim() == 3:
        height_map = height_map.unsqueeze(0)

    # Compute gradients using Sobel operator
    grad_y = F.conv2d(
        F.pad(height_map, (1, 1, 1, 1), mode='replicate'),
        torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]).to(height_map),
        padding=0
    )

    grad_x = F.conv2d(
        F.pad(height_map, (1, 1, 1, 1), mode='replicate'),
        torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).to(height_map),
        padding=0
    )

    # Scale gradients
    grad_x = grad_x * scale
    grad_y = grad_y * scale

    # Create normal map with swapped and inverted axes
    normal_map = torch.cat([
        grad_y,  # Y axis (swapped from X)
        grad_x, # X axis (swapped from Y)
        torch.ones_like(grad_x)
    ], dim=1)

    # Normalize vectors
    normal_map = F.normalize(normal_map, dim=1)

    return normal_map[0]


def compute_normal_scharr(image_rgb, scale_xy=1.0, fix_black=True):
    """
    Compute normal map using Scharr operators on an RGB image.
    Matches the WSL NormalMapSimple + ConvertNormals pipeline exactly.

    Args:
        image_rgb: numpy array (H, W, 3) uint8 RGB image
        scale_xy: float, scaling factor for XY gradients
        fix_black: bool, fix near-black artifacts in the normal map

    Returns:
        numpy array: normal map (H, W, 3) in range [0, 255] uint8
    """
    t = image_rgb.astype(np.float32) / 255.0
    L = np.mean(t[:, :, :3], axis=2)

    t[:, :, 0] = cv2.Scharr(L, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_REFLECT) * -1
    t[:, :, 1] = cv2.Scharr(L, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REFLECT) * -1
    t[:, :, 2] = 1.0

    t_tensor = torch.from_numpy(t).unsqueeze(0)
    t_tensor[:, :, :, :2] *= scale_xy
    t_tensor[:, :, :, :3] = F.normalize(t_tensor[:, :, :, :3], dim=3) / 2 + 0.5

    if fix_black:
        key = torch.clamp(1 - t_tensor[:, :, :, 2] * 2, min=0, max=1)
        t_tensor[:, :, :, 0] += key * 0.5
        t_tensor[:, :, :, 1] += key * 0.5
        t_tensor[:, :, :, 2] += key

    t_norm = t_tensor[:, :, :, :3] * 2 - 1
    lengths = torch.clamp(torch.sqrt(torch.sum(t_norm ** 2, dim=3, keepdim=True)), min=1e-6)
    t_norm = t_norm / lengths
    t_tensor[:, :, :, :3] = (t_norm + 1) / 2

    result = (t_tensor[0].numpy() * 255).clip(0, 255).astype(np.uint8)
    return result


def infer_depth_pipe(pipe, test_image, task_name, device, optimize=True):
    """
    Run inference on a single pipeline

    Args:
        pipe: PBRFusion4 pipeline
        test_image: numpy array or PIL Image
        task_name: 'depth' or 'normal'
        device: torch device
        optimize: if True, limit processing to 2048x2048 and resize back to original

    Returns:
        PIL Image: output image
    """
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(pipe.device.type)

    with autocast_ctx:
        # Prepare image
        if isinstance(test_image, Image.Image):
            test_image = np.array(test_image)

        # Store original dimensions for resizing back
        original_height, original_width = test_image.shape[:2]

        test_image = test_image.astype(np.float32)

        # Optionally resize for optimized processing
        if optimize:
            if max(test_image.shape[:2]) > 1024:
                scale = 1024 / max(test_image.shape[:2])
            elif min(test_image.shape[:2]) < 384:
                scale = 384 / min(test_image.shape[:2])
            else:
                scale = 1.0

            if scale != 1.0:
                new_shape = (int(test_image.shape[1] * scale), int(test_image.shape[0] * scale))
                test_image = cv2.resize(test_image, new_shape)

        test_image = test_image.astype(np.float16)

        # Convert to tensor
        test_image = torch.tensor(test_image).permute(2, 0, 1).unsqueeze(0)
        test_image = test_image / 127.5 - 1.0
        test_image = test_image.to(device)

        # Task embedding - sin/cos encoding as per original Lotus implementation
        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

        # Run inference (deterministic, no generator needed)
        pred = pipe(
            rgb_in=test_image,
            prompt='',
            num_inference_steps=1,
            output_type='np',
            timesteps=[999],
            task_emb=task_emb,
        ).images[0]

        # Post-process
        if task_name == 'depth':
            output_npy = pred.mean(axis=-1)
            depth_normalized = ((output_npy - output_npy.min()) / (output_npy.max() - output_npy.min()) * 255).astype(np.uint8)
            output_color = Image.fromarray(depth_normalized, mode='L')
        else:
            output_npy = pred
            output_color = Image.fromarray((output_npy * 255).astype(np.uint8))

        # Resize back to original dimensions if we scaled down
        if optimize and (output_color.width != original_width or output_color.height != original_height):
            output_color = output_color.resize((original_width, original_height), Image.LANCZOS)

    return output_color


class PBRFusion4SimpleDepthAndNormalGenerator:
    """
    Generate depth maps and normal maps from images using PBRFusion4 (simple version without seamless processing)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity_blend": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "opengl_normal": ("BOOLEAN", {
                    "default": False,
                }),
                "optimize": ("BOOLEAN", {
                    "default": True,
                }),
                "bilateral_d": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
                "bilateral_sigma_color": ("FLOAT", {
                    "default":50.0,
                    "min": 1.0,
                    "max": 200.0,
                    "step": 1.0,
                }),
                "bilateral_sigma_space": ("FLOAT", {
                    "default": 50.0,
                    "min": 1.0,
                    "max": 200.0,
                    "step": 1.0,
                }),
                "scharr_scale_xy": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("depth", "depth_filtered", "normal", "normal_scharr", "intensity")
    FUNCTION = "generate"
    CATEGORY = "PBRFUSION4/Generation"
    DESCRIPTION = """
Generates depth map and normal map from input image using PBRFusion4 diffusion model.
Simple version without seamless/tileable processing.
- Deterministic depth estimation (no seed needed)
- Depth map optimized for PBR workflows
- Normal map generated from depth using Sobel operators
- Scharr normal map output for higher precision gradients
- Supports intensity blending for better detail preservation
"""

    def generate(self, image, intensity_blend, opengl_normal, optimize, bilateral_d, bilateral_sigma_color, bilateral_sigma_space, scharr_scale_xy):
        """
        Generate depth and normal maps

        Args:
            image: Input image tensor (B, H, W, C)
            intensity_blend: Factor for blending original intensity
            directx_normal: If True, invert Y/green channel for DirectX format
            optimize: If True, process at 1024x1024 max and resize back to original
            bilateral_d: Bilateral filter diameter
            bilateral_sigma_color: Bilateral filter sigma in color space
            bilateral_sigma_space: Bilateral filter sigma in coordinate space

        Returns:
            tuple: (depth_image, depth_filtered_image, normal_image, intensity_image)
        """
        device = get_device()

        # Load models
        pipe_g, pipe_d = load_pbrfusion4_models('depth', device)

        # Process first image in batch
        input_tensor = image[0]  # (H, W, C)

        # Convert to numpy and PIL
        input_array = (input_tensor.cpu().numpy() * 255).astype(np.uint8)
        input_pil = Image.fromarray(input_array)

        # === DEPTH PROCESSING ===

        # Get the depth map
        depth_pil = infer_depth_pipe(pipe_d, input_pil, 'depth', device, optimize)
        processed_depth = np.array(depth_pil)

        # Normalize depth
        depth_normalized = ((processed_depth - processed_depth.min()) / (processed_depth.max() - processed_depth.min()) * 255).astype(np.uint8)

        # === BILATERAL FILTER FOR DEPTH ===
        depth_filtered = cv2.bilateralFilter(
            depth_normalized,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space
        )

        # Extract intensity map from original input
        intensity_map = get_image_intensity(input_array, gamma_correction=1.0)

        # === NORMAL MAP GENERATION ===
        # Create depth+intensity blend for normal map generation only (using filtered depth)
        if intensity_blend > 0:
            # Convert filtered depth to RGB for blending
            depth_rgb = cv2.cvtColor(depth_filtered, cv2.COLOR_GRAY2RGB)
            # Blend depth with intensity
            depth_rgb = blend_numpy_images(depth_rgb, intensity_map, blend_factor=intensity_blend)
            # Convert back to grayscale for normal generation
            depth_for_normal = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2GRAY)
        else:
            depth_for_normal = depth_filtered

        # Generate normal from depth
        depth_tensor = torch.from_numpy(depth_for_normal).to(device).float() / 255.0
        normal_map = compute_normal_from_height(depth_tensor, scale=2.0)

        # Convert to display format (H, W, C) [0, 1]
        normal_display = normal_map.cpu().numpy().transpose(1, 2, 0)
        normal_display = np.clip(normal_display, -1, 1)
        normal_display = ((normal_display + 1.0) * 0.5 * 255).astype(np.uint8)

        # OpenGL normal has Y pointing up (default from Sobel).
        # When opengl_normal is False, flip Y for DirectX format.
        if not opengl_normal:
            normal_display[:, :, 1] = 255 - normal_display[:, :, 1]

        # Convert to tensor for output
        normal_tensor = torch.from_numpy(normal_display).float() / 255.0
        normal_tensor = normal_tensor.unsqueeze(0)  # Add batch dimension

        # === SCHARR NORMAL MAP (bilateral filtered depth → blend → Scharr → fix_black → normalize) ===
        depth_blurred_rgb = cv2.cvtColor(depth_filtered, cv2.COLOR_GRAY2RGB)
        if intensity_blend > 0:
            scharr_input = blend_numpy_images(depth_blurred_rgb, intensity_map, blend_factor=intensity_blend)
        else:
            scharr_input = depth_blurred_rgb

        scharr_normal = compute_normal_scharr(scharr_input, scale_xy=scharr_scale_xy)

        if not opengl_normal:
            scharr_normal[:, :, 1] = 255 - scharr_normal[:, :, 1]

        scharr_normal_tensor = torch.from_numpy(scharr_normal).float() / 255.0
        scharr_normal_tensor = scharr_normal_tensor.unsqueeze(0)

        # Convert depth to RGB format for output
        depth_array = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)

        # Convert depth to tensor
        depth_tensor = torch.from_numpy(depth_array).float() / 255.0
        depth_tensor = depth_tensor.unsqueeze(0)  # Add batch dimension

        # Convert filtered depth to RGB and tensor
        depth_filtered_array = cv2.cvtColor(depth_filtered, cv2.COLOR_GRAY2RGB)
        depth_filtered_tensor = torch.from_numpy(depth_filtered_array).float() / 255.0
        depth_filtered_tensor = depth_filtered_tensor.unsqueeze(0)  # Add batch dimension

        # Convert intensity map to tensor (already in RGB format from get_image_intensity)
        intensity_tensor = torch.from_numpy(intensity_map).float() / 255.0
        intensity_tensor = intensity_tensor.unsqueeze(0)  # Add batch dimension

        return (depth_tensor, depth_filtered_tensor, normal_tensor, scharr_normal_tensor, intensity_tensor)


class NormalMapFlipY:
    """
    Flip Y axis (green channel) of normal map to convert between OpenGL and DirectX formats
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "flip_y"
    CATEGORY = "PBRFUSION4/Utils"
    DESCRIPTION = "Flips the Y axis (green channel) of a normal map to convert between OpenGL and DirectX formats."

    def flip_y(self, normal_map):
        # Clone to avoid modifying original
        result = normal_map.clone()
        # Flip green channel (index 1): 1.0 - value
        result[:, :, :, 1] = 1.0 - result[:, :, :, 1]
        return (result,)


class BlackThreshold:
    """
    Threshold near-black pixels to pure black to remove artifacts
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_threshold"
    CATEGORY = "PBRFUSION4/Utils"
    DESCRIPTION = "Turns all pixels below the threshold to pure black. Useful for cleaning up artifacts in near-black areas after upscaling."

    def apply_threshold(self, image, threshold):
        # Clone to avoid modifying original
        result = image.clone()
        # Create mask for pixels where all RGB channels are below threshold
        mask = (result[:, :, :, 0] < threshold) & (result[:, :, :, 1] < threshold) & (result[:, :, :, 2] < threshold)
        # Set all channels to 0 where mask is True
        result[mask] = 0.0
        return (result,)


class SmartUpscaleCalculator:
    """
    Calculate target resolution and upscale factor based on input image and target size
    """

    RESOLUTION_MAP = {
        "1k": 1024,
        "2k": 2048,
        "4k": 4096,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_resolution": (["1k", "2k", "4k"], {
                    "default": "2k",
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("width", "height", "scale_factor", "needs_upscale")
    FUNCTION = "calculate"
    CATEGORY = "PBRFUSION4/Utils"
    DESCRIPTION = "Detects input image dimensions and calculates target width, height, and upscale factor for the target resolution (1k/2k/4k). Preserves aspect ratio. Scale capped at 4x. Outputs needs_upscale=True if scale > 1."

    def calculate(self, image, target_resolution):
        # Get input dimensions from image (B, H, W, C)
        height = image.shape[1]
        width = image.shape[2]

        # Get target max dimension
        target_max = self.RESOLUTION_MAP[target_resolution]

        # Find the max side of input
        input_max = max(width, height)

        # Calculate scale factor
        scale_factor = target_max / input_max

        # Cap scale factor at 4.0 (max practical upscale)
        if scale_factor > 4.0:
            scale_factor = 4.0

        # Calculate output dimensions
        output_width = int(width * scale_factor)
        output_height = int(height * scale_factor)

        # Determine if upscaling is needed
        needs_upscale = scale_factor > 1.0

        return (output_width, output_height, scale_factor, needs_upscale)


class ClampResolution:
    """
    Clamp resolution to max size while preserving aspect ratio
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "max_size": ("INT", {"default": 2048, "min": 64, "max": 8192}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("width", "height", "was_clamped")
    FUNCTION = "clamp"
    CATEGORY = "PBRFUSION4/Utils"
    DESCRIPTION = "If either width or height exceeds max_size, scales both dimensions down to fit within max_size while preserving aspect ratio. Outputs was_clamped=True if clamping occurred."

    def clamp(self, width, height, max_size):
        # Check if either dimension exceeds max
        if width <= max_size and height <= max_size:
            return (width, height, False)

        # Find the larger dimension and calculate scale
        if width >= height:
            scale = max_size / width
        else:
            scale = max_size / height

        # Apply scale
        output_width = int(width * scale)
        output_height = int(height * scale)

        return (output_width, output_height, True)


class ConditionalUpscale:
    """
    Upscale image using model with bypass option
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "PBRFUSION4/Utils"
    DESCRIPTION = "Upscales image using model if enable is True, otherwise passes through unchanged."

    def upscale(self, upscale_model, image, enable):
        if not enable:
            return (image,)

        device = mm.get_torch_device()

        memory_required = mm.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        mm.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 512
        overlap = 32

        oom = True
        try:
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                    oom = False
                except mm.OOM_EXCEPTION as e:
                    tile //= 2
                    if tile < 128:
                        raise e
        finally:
            upscale_model.to("cpu")

        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AIXPOLY_PBRFusion4SimpleDepthAndNormalGenerator": PBRFusion4SimpleDepthAndNormalGenerator,
    "AIXPOLY_NormalMapFlipY": NormalMapFlipY,
    "AIXPOLY_BlackThreshold": BlackThreshold,
    "AIXPOLY_SmartUpscaleCalculator": SmartUpscaleCalculator,
    "AIXPOLY_ClampResolution": ClampResolution,
    "AIXPOLY_ConditionalUpscale": ConditionalUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIXPOLY_PBRFusion4SimpleDepthAndNormalGenerator": "PBRFusion4 Simple Depth & Normal Generator",
    "AIXPOLY_NormalMapFlipY": "Normal Map Flip Y (DirectX/OpenGL)",
    "AIXPOLY_BlackThreshold": "Black Threshold Filter",
    "AIXPOLY_SmartUpscaleCalculator": "Smart Upscale Calculator",
    "AIXPOLY_ClampResolution": "Clamp Resolution",
    "AIXPOLY_ConditionalUpscale": "Conditional Upscale (using Model)",
}
