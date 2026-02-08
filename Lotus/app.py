from gradio_imageslider import ImageSlider
import functools
import os
import tempfile
import diffusers
import gradio as gr
import imageio as imageio
import numpy as np
import spaces
import torch as torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from pathlib import Path
import gradio
from gradio.utils import get_cache_folder
from infer import lotus, lotus_video
import transformers
from huggingface_hub import login
import cv2

transformers.utils.move_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "HF_TOKEN_LOGIN" in os.environ:
    login(token=os.environ["HF_TOKEN_LOGIN"])

def apply_gaussian_blur(image, radius=1.0):
    """Apply Gaussian blur to PIL Image with specified radius"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

class NormalMapSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale_XY": ("FLOAT",{"default": 1, "min": 0, "max": 100, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normal_map"

    CATEGORY = "image/filters"

    def normal_map(self, images, scale_XY):
        t = images.detach().clone().cpu().numpy().astype(np.float32)
        L = np.mean(t[:,:,:,:3], axis=3)
        for i in range(t.shape[0]):
            t[i,:,:,0] = cv2.Scharr(L[i], -1, 1, 0, cv2.BORDER_REFLECT) * -1
            t[i,:,:,1] = cv2.Scharr(L[i], -1, 0, 1, cv2.BORDER_REFLECT)
        t[:,:,:,2] = 1
        t = torch.from_numpy(t)
        t[:,:,:,:2] *= scale_XY
        t[:,:,:,:3] = torch.nn.functional.normalize(t[:,:,:,:3], dim=3) / 2 + 0.5
        return (t,)

class ConvertNormals:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normals": ("IMAGE",),
                "input_mode": (["BAE", "MiDaS", "Standard", "DirectX"],),
                "output_mode": (["BAE", "MiDaS", "Standard", "DirectX"],),
                "scale_XY": ("FLOAT",{"default": 1, "min": 0, "max": 100, "step": 0.001}),
                "normalize": ("BOOLEAN", {"default": True}),
                "fix_black": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "optional_fill": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_normals"

    CATEGORY = "image/filters"

    def convert_normals(self, normals, input_mode, output_mode, scale_XY, normalize, fix_black, optional_fill=None):
        try:
            t = normals.detach().clone()

            if input_mode == "BAE":
                t[:,:,:,0] = 1 - t[:,:,:,0] # invert R
            elif input_mode == "MiDaS":
                t[:,:,:,:3] = torch.stack([1 - t[:,:,:,2], t[:,:,:,1], t[:,:,:,0]], dim=3) # BGR -> RGB and invert R
            elif input_mode == "DirectX":
                t[:,:,:,1] = 1 - t[:,:,:,1] # invert G

            if fix_black:
                key = torch.clamp(1 - t[:,:,:,2] * 2, min=0, max=1)
                if optional_fill is None:
                    t[:,:,:,0] += key * 0.5
                    t[:,:,:,1] += key * 0.5
                    t[:,:,:,2] += key
                else:
                    fill = optional_fill.detach().clone()
                    if fill.shape[1:3] != t.shape[1:3]:
                        fill = torch.nn.functional.interpolate(fill.movedim(-1,1), size=(t.shape[1], t.shape[2]), mode='bilinear').movedim(1,-1)
                    if fill.shape[0] != t.shape[0]:
                        fill = fill[0].unsqueeze(0).expand(t.shape[0], -1, -1, -1)
                    t[:,:,:,:3] += fill[:,:,:,:3] * key.unsqueeze(3).expand(-1, -1, -1, 3)

            t[:,:,:,:2] = (t[:,:,:,:2] - 0.5) * scale_XY + 0.5

            if normalize:
                # Transform to [-1, 1] range
                t_norm = t[:,:,:,:3] * 2 - 1

                # Calculate the length of each vector
                lengths = torch.sqrt(torch.sum(t_norm**2, dim=3, keepdim=True))

                # Avoid division by zero
                lengths = torch.clamp(lengths, min=1e-6)

                # Normalize each vector to unit length
                t_norm = t_norm / lengths

                # Transform back to [0, 1] range
                t[:,:,:,:3] = (t_norm + 1) / 2

            if output_mode == "BAE":
                t[:,:,:,0] = 1 - t[:,:,:,0] # invert R
            elif output_mode == "MiDaS":
                t[:,:,:,:3] = torch.stack([t[:,:,:,2], t[:,:,:,1], 1 - t[:,:,:,0]], dim=3) # invert R and BGR -> RGB
            elif output_mode == "DirectX":
                t[:,:,:,1] = 1 - t[:,:,:,1] # invert G

            return (t,)
        except Exception as e:
            print(f"Error in convert_normals: {str(e)}")
            return (normals,)

def get_image_intensity(img, gamma_correction=1.0):
    """
    Extract intensity map from an image using HSV color space
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

def blend_numpy_images(image1, image2, blend_factor=0.25, mode="normal"):
    """
    Blend two numpy images using normal mode
    """
    # Convert to float32 and normalize to 0-1
    img1 = image1.astype(np.float32) / 255.0
    img2 = image2.astype(np.float32) / 255.0

    # Normal blend mode
    blended = img1 * (1 - blend_factor) + img2 * blend_factor

    # Convert back to uint8
    blended = (blended * 255.0).clip(0, 255).astype(np.uint8)
    return blended

def process_normal_map(image):
    """
    Process image through NormalMapSimple and ConvertNormals
    """
    # Convert numpy image to torch tensor with batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0

    # Create instances of the classes
    normal_map_generator = NormalMapSimple()
    normal_converter = ConvertNormals()

    # Generate initial normal map
    normal_map = normal_map_generator.normal_map(image_tensor, scale_XY=1.0)[0]

    # Convert normal map from Standard to Standard (OpenGL)
    converted_normal = normal_converter.convert_normals(
        normal_map,
        input_mode="Standard",
        output_mode="Standard",
        scale_XY=1.0,
        normalize=True,
        fix_black=True
    )[0]

    # Convert back to numpy array
    result = (converted_normal.squeeze(0).numpy() * 255).astype(np.uint8)
    return result

def infer(path_input, seed=None):
    name_base, name_ext = os.path.splitext(os.path.basename(path_input))
    _, output_d = lotus(path_input, 'depth', seed, device)

    # Apply Gaussian blur with 0.75 radius
    output_d = apply_gaussian_blur(output_d, radius=0.75)

    # Convert depth to numpy for normal map processing
    depth_array = np.array(output_d)

    # Load original image for intensity blending
    input_image = Image.open(path_input)
    input_array = np.array(input_image)

    # Get intensity map from original image
    intensity_map = get_image_intensity(input_array, gamma_correction=1.0)

    # Resize intensity_map to match depth_array dimensions
    depth_height, depth_width = depth_array.shape[:2]
    if intensity_map.shape[:2] != (depth_height, depth_width):
        intensity_map = cv2.resize(intensity_map, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)

    # Blend depth with intensity map
    blended_result = blend_numpy_images(
        cv2.cvtColor(depth_array, cv2.COLOR_RGB2BGR if len(depth_array.shape) == 3 else cv2.COLOR_GRAY2BGR),
        intensity_map,
        blend_factor=0.15,
        mode="normal"
    )

    # Generate normal map from blended result
    normal_map = process_normal_map(blended_result)

    if not os.path.exists("files/output"):
        os.makedirs("files/output")
    d_save_path = os.path.join("files/output", f"{name_base}_d{name_ext}")
    n_save_path = os.path.join("files/output", f"{name_base}_n{name_ext}")

    output_d.save(d_save_path)
    Image.fromarray(normal_map).save(n_save_path)

    return [path_input, d_save_path], [path_input, n_save_path]

def infer_video(path_input, seed=None):
    _, frames_d, fps = lotus_video(path_input, 'depth', seed, device)
    
    # Apply Gaussian blur to each frame
    blurred_frames = []
    for frame in frames_d:
        # Convert numpy array to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        # Apply blur and convert back to numpy array
        blurred_frame = apply_gaussian_blur(frame_pil, radius=0.75)
        blurred_frames.append(np.array(blurred_frame))
    
    if not os.path.exists("files/output"):
        os.makedirs("files/output")
    name_base, _ = os.path.splitext(os.path.basename(path_input))
    d_save_path = os.path.join("files/output", f"{name_base}_d.mp4")
    imageio.mimsave(d_save_path, blurred_frames, fps=fps)
    return d_save_path

def run_demo_server():
    infer_gpu = spaces.GPU(functools.partial(infer))
    infer_video_gpu = spaces.GPU(functools.partial(infer_video))
    gradio_theme = gr.themes.Default()

    with gr.Blocks(
        theme=gradio_theme,
        title="LOTUS (Depth & Normal Maps - Discriminative)",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
            h1 {
                text-align: center;
                display: block;
            }
            h2 {
                text-align: center;
                display: block;
            }
            h3 {
                text-align: center;
                display: block;
            }
            .md_feedback li {
                margin-bottom: 0px !important;
            }
        """,
        head="""
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """,
    ) as demo:
        with gr.Tabs(elem_classes=["tabs"]):
            with gr.Tab("IMAGE"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Input Image",
                            type="filepath",
                        )
                        with gr.Row():
                            image_submit_btn = gr.Button(
                                value="Predict Depth!", variant="primary"
                            )
                            image_reset_btn = gr.Button(value="Reset")
                    with gr.Column():
                        image_output_d = ImageSlider(
                            label="Depth Output (Discriminative)",
                            type="filepath",
                            interactive=False,
                            elem_classes="slider",
                            position=0.25,
                        )
                        image_output_n = ImageSlider(
                            label="OpenGL Normal Map Output",
                            type="filepath",
                            interactive=False,
                            elem_classes="slider",
                            position=0.25,
                        )

                gr.Examples(
                    fn=infer_gpu,
                    examples=sorted([
                        [os.path.join("files", "images", name)]
                        for name in os.listdir(os.path.join("files", "images"))
                    ]),
                    inputs=[image_input],
                    outputs=[image_output_d, image_output_n],
                    cache_examples=False,
                )

            with gr.Tab("VIDEO"):
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(
                            label="Input Video",
                            autoplay=True,
                            loop=True,
                        )
                        with gr.Row():
                            video_submit_btn = gr.Button(
                                value="Predict Depth!", variant="primary"
                            )
                            video_reset_btn = gr.Button(value="Reset")
                    with gr.Column():
                        video_output_d = gr.Video(
                            label="Depth Output (Discriminative)",
                            interactive=False,
                            autoplay=True,
                            loop=True,
                            show_share_button=True,
                        )

                gr.Examples(
                    fn=infer_video_gpu,
                    examples=sorted([
                        [os.path.join("files", "videos", name)]
                        for name in os.listdir(os.path.join("files", "videos"))
                    ]),
                    inputs=[input_video],
                    outputs=[video_output_d],
                    cache_examples=False,
                )

        ### Image
        image_submit_btn.click(
            fn=infer_gpu,
            inputs=[image_input],
            outputs=[image_output_d, image_output_n],
            concurrency_limit=1,
        )
        image_reset_btn.click(
            fn=lambda: [None, None],
            inputs=[],
            outputs=[image_output_d, image_output_n],
            queue=False,
        )

        ### Video
        video_submit_btn.click(
            fn=infer_video_gpu,
            inputs=[input_video],
            outputs=[video_output_d],
            queue=True,
        )
        video_reset_btn.click(
            fn=lambda: None,
            inputs=[],
            outputs=[video_output_d],
        )

        ### Server launch
        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
        )

def main():
    os.system("pip freeze")
    if os.path.exists("files/output"):
        os.system("rm -rf files/output")
    run_demo_server()

if __name__ == "__main__":
    main()