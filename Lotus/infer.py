# from utils.args import parse_args
import logging
import os
import argparse
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.utils import check_min_version

from pipeline import LotusGPipeline, LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.seed_all import seed_all

from contextlib import nullcontext
import cv2

check_min_version('0.28.0.dev0')

def infer_pipe(pipe, test_image, task_name, seed, device, video_depth=False):
    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(seed)

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(pipe.device.type)
    with autocast_ctx:

        if video_depth == False:
            test_image = Image.open(test_image).convert('RGB')
        test_image = np.array(test_image).astype(np.float32)
        if max(test_image.shape[:2]) > 1024:
            # resize for a maximum size of 1024
            scale = 1024 / max(test_image.shape[:2])
        elif min(test_image.shape[:2]) < 384:
            # resize for a minimum size of 384
            scale = 384 / min(test_image.shape[:2])
        else:
            scale = 1.0
        new_shape = (int(test_image.shape[1] * scale), int(test_image.shape[0] * scale))
        test_image = cv2.resize(test_image, new_shape)
        test_image = test_image.astype(np.float16)
        test_image = torch.tensor(test_image).permute(2,0,1).unsqueeze(0)
        test_image = test_image / 127.5 - 1.0 
        test_image = test_image.to(device)

        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

        # Run
        pred = pipe(
            rgb_in=test_image, 
            prompt='', 
            num_inference_steps=1, 
            generator=generator, 
            # guidance_scale=0,
            output_type='np',
            timesteps=[999],
            task_emb=task_emb,
            ).images[0]

        # NEW CODE:
        if task_name == 'depth':
            output_npy = pred.mean(axis=-1)
            depth_normalized = ((output_npy - output_npy.min()) / (output_npy.max() - output_npy.min()) * 255).astype(np.uint8)
            output_color = Image.fromarray(depth_normalized, mode='L')
        else:
            output_npy = pred
            output_color = Image.fromarray((output_npy * 255).astype(np.uint8))

    return output_color

def infer_pipe_video(pipe, test_image, task_name, generator, device, latents=None):
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(pipe.device.type)
    with autocast_ctx:
        test_image = np.array(test_image).astype(np.float16)
        test_image = torch.tensor(test_image).permute(2,0,1).unsqueeze(0)
        test_image = test_image / 127.5 - 1.0 
        test_image = test_image.to(device)

        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

        # Run
        output = pipe(
            rgb_in=test_image, 
            prompt='', 
            num_inference_steps=1, 
            generator=generator, 
            latents=latents,
            # guidance_scale=0,
            output_type='np',
            timesteps=[999],
            task_emb=task_emb,
            return_dict=False
            )
        pred = output[0][0]
        last_frame_latent = output[2]

        # Post-process the prediction
        if task_name == 'depth':
            output_npy = pred.mean(axis=-1)
            output_color = colorize_depth_map(output_npy, reverse_color=True)
        else:
            output_npy = pred
            output_color = Image.fromarray((output_npy * 255).astype(np.uint8))

    return output_color, last_frame_latent

def load_pipe(task_name, device):
    if task_name == 'depth':
        model_g = 'jingheya/lotus-depth-g-v2-1-disparity'
        model_d = 'NightRaven109/LotusDInstNorm'
    else:
        model_g = 'jingheya/lotus-normal-g-v1-0'
        model_d = 'jingheya/lotus-normal-d-v1-0'

    dtype = torch.float16
    pipe_g = LotusGPipeline.from_pretrained(
        model_g,
        torch_dtype=dtype,
    )
    pipe_d = LotusDPipeline.from_pretrained(
        model_d,
        torch_dtype=dtype,
    )
    pipe_g.to(device)
    pipe_d.to(device)
    pipe_g.set_progress_bar_config(disable=True)
    pipe_d.set_progress_bar_config(disable=True)
    logging.info(f"Successfully loading pipeline from {model_g} and {model_d}.")
    return pipe_g, pipe_d

def lotus_video(input_video, task_name, seed, device):
    pipe_g, pipe_d = load_pipe(task_name, device)
    
    # load the video and split it into frames
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # generate latents_common for lotus-g
    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    last_frame_latent = None
    latent_common = torch.randn(
        (1, 4, height // pipe_g.vae_scale_factor, width // pipe_g.vae_scale_factor), generator=generator, dtype=pipe_g.dtype, device=device
    )

    output_g = []
    output_d = []
    for frame in frames:
        latents = latent_common
        if last_frame_latent is not None:
            latents = 0.9 * latents + 0.1 * last_frame_latent
        output_frame_g, last_frame_latent = infer_pipe_video(pipe_g, frame, task_name, seed, device, latents)
        output_frame_d = infer_pipe(pipe_d, frame, task_name, seed, device, video_depth=True)
        output_g.append(output_frame_g)
        output_d.append(output_frame_d)

    return output_g, output_d, fps

def lotus(image_input, task_name, seed, device):
    pipe_g, pipe_d = load_pipe(task_name, device)
    output_g = infer_pipe(pipe_g, image_input, task_name, seed, device)
    output_d = infer_pipe(pipe_d, image_input, task_name, seed, device)
    return output_g, output_d

def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run Lotus..."
    )
    # model settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The used prediction_type. ",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="regression", # "generation"
        help="Whether to use the generation or regression pipeline."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="depth", # "normal"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    # inference settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory."
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    
    args = parser.parse_args()

    return args

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run inference...")
    
    args = parse_args()

    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output dir = {args.output_dir}")

    output_dir_color = os.path.join(args.output_dir, f'{args.task_name}_vis')
    output_dir_npy = os.path.join(args.output_dir, f'{args.task_name}')
    if not os.path.exists(output_dir_color): os.makedirs(output_dir_color)
    if not os.path.exists(output_dir_npy): os.makedirs(output_dir_npy)

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float16

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Data --------------------
    root_dir = Path(args.input_dir)
    test_images = list(root_dir.rglob('*.png')) + list(root_dir.rglob('*.jpg'))
    test_images = sorted(test_images)
    print('==> There are', len(test_images), 'images for validation.')
    # -------------------- Model --------------------
    
    if args.mode == 'generation':
        pipeline = LotusGPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
    elif args.mode == 'regression':
        pipeline = LotusDPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
    logging.info(f"Successfully loading pipeline from {args.pretrained_model_name_or_path}.")

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()


    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i in tqdm(range(len(test_images))):
            # Preprocess validation image
            test_image = Image.open(test_images[i]).convert('RGB')
            test_image = np.array(test_image).astype(np.float16)
            test_image = torch.tensor(test_image).permute(2,0,1).unsqueeze(0)
            test_image = test_image / 127.5 - 1.0 
            test_image = test_image.to(device)

            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

            # Run
            pred = pipeline(
                rgb_in=test_image, 
                prompt='', 
                num_inference_steps=1, 
                generator=generator, 
                # guidance_scale=0,
                output_type='np',
                timesteps=[args.timestep],
                task_emb=task_emb,
                ).images[0]

            # Post-process the prediction
            save_file_name = os.path.basename(test_images[i])[:-4]
            if args.task_name == 'depth':
                output_npy = pred.mean(axis=-1)
                output_color = colorize_depth_map(output_npy)
            else:
                output_npy = pred
                output_color = Image.fromarray((output_npy * 255).astype(np.uint8))

            output_color.save(os.path.join(output_dir_color, f'{save_file_name}.png'))
            np.save(os.path.join(output_dir_npy, f'{save_file_name}.npy'), output_npy)
    
    print('==> Inference is done. \n==> Results saved to:', args.output_dir)

if __name__ == '__main__':
    main()
