import os
import torch
import gradio as gr
# from PIL import Image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize global variables for models and pipeline
text_encoder = None
tokenizer = None
vae = None
scheduler = None
unet = None
pipe = None

def load_models():
    global text_encoder, tokenizer, vae, scheduler, unet, pipe

    if text_encoder is None:
        ckpt_dir = f'{root_dir}/weights/Kolors'

        # Load the text encoder on CPU (this speeds stuff up 2x)
        text_encoder = ChatGLMModel.from_pretrained(
            f'{ckpt_dir}/text_encoder',
            torch_dtype=torch.float16).to('cpu').half()
        tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')

        # Load the VAE and UNet on GPU
        vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half().to('cuda')
        scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
        unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half().to('cuda')

        # Prepare the pipeline
        pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False)
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()  # Enable offloading to balance CPU/GPU usage

def infer(prompt, use_random_seed, seed, height, width, num_inference_steps, guidance_scale, num_images_per_prompt):
    load_models()

    if use_random_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator(pipe.device).manual_seed(seed)
    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator
    ).images

    saved_images = []
    output_dir = f'{root_dir}/scripts/outputs'
    os.makedirs(output_dir, exist_ok=True)

    for i, image in enumerate(images):
        file_path = os.path.join(output_dir, 'sample_test.jpg')
        base_name, ext = os.path.splitext(file_path)
        counter = 1
        while os.path.exists(file_path):
            file_path = f"{base_name}_{counter}{ext}"
            counter += 1
        image.save(file_path)
        saved_images.append(file_path)

    return saved_images

def update_dimensions(aspect_ratio):
    """Update height and width based on selected aspect ratio"""
    # Base resolution to maintain approximately similar pixel count
    base_resolution = 1024  # Already divisible by 8
    
    def round_to_multiple_of_8(value):
        """Round the value to the nearest multiple of 8"""
        return round(value / 8) * 8
    
    if aspect_ratio == "1:1":
        return gr.update(value=base_resolution), gr.update(value=base_resolution)
    elif aspect_ratio == "9:16": 
        # Portrait orientation
        width = round_to_multiple_of_8(base_resolution * 0.75)  # ~768
        height = round_to_multiple_of_8(width * 16/9)           # ~1360
        return gr.update(value=height), gr.update(value=width)
    elif aspect_ratio == "16:9":
        # Landscape orientation
        height = round_to_multiple_of_8(base_resolution * 0.75)  # ~768
        width = round_to_multiple_of_8(height * 16/9)            # ~1360
        return gr.update(value=height), gr.update(value=width)
    elif aspect_ratio == "3:4":
        # Portrait orientation
        width = round_to_multiple_of_8(base_resolution * 0.9)   # ~920
        height = round_to_multiple_of_8(width * 4/3)            # ~1224
        return gr.update(value=height), gr.update(value=width)
    elif aspect_ratio == "4:3":
        # Landscape orientation
        height = round_to_multiple_of_8(base_resolution * 0.9)  # ~920
        width = round_to_multiple_of_8(height * 4/3)            # ~1224
        return gr.update(value=height), gr.update(value=width)
    elif aspect_ratio == "2:3":
        # Portrait orientation
        width = round_to_multiple_of_8(base_resolution * 0.85)  # ~872
        height = round_to_multiple_of_8(width * 3/2)            # ~1304
        return gr.update(value=height), gr.update(value=width)
    elif aspect_ratio == "3:2":
        # Landscape orientation
        height = round_to_multiple_of_8(base_resolution * 0.85)  # ~872
        width = round_to_multiple_of_8(height * 3/2)             # ~1304
        return gr.update(value=height), gr.update(value=width)
    elif aspect_ratio == "21:9":
        # Ultra-wide landscape
        height = round_to_multiple_of_8(base_resolution * 0.65)  # ~664
        width = round_to_multiple_of_8(height * 21/9)            # ~1552
        return gr.update(value=height), gr.update(value=width)
    else:
        return gr.update(), gr.update()

def gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Kolors: Diffusion Model Gradio Interface")
                prompt = gr.Textbox(label="Prompt")
                use_random_seed = gr.Checkbox(label="Use Random Seed", value=True)
                seed = gr.Slider(minimum=0, maximum=2**32 - 1, step=1, label="Seed", randomize=True, visible=False)
                use_random_seed.change(lambda x: gr.update(visible=not x), use_random_seed, seed)
                
                # Add aspect ratio radio buttons
                aspect_ratio = gr.Radio(
                    choices=["1:1", "9:16", "16:9", "3:4", "4:3", "2:3", "3:2", "21:9"], 
                    label="Aspect Ratio", 
                    value="1:1"
                )
                
                height = gr.Slider(minimum=128, maximum=2048, step=8, label="Height", value=1024)
                width = gr.Slider(minimum=128, maximum=2048, step=8, label="Width", value=1024)
                
                # Connect aspect ratio change to dimension updates
                aspect_ratio.change(
                    fn=update_dimensions,
                    inputs=aspect_ratio,
                    outputs=[height, width]
                )
                
                num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, label="Inference Steps", value=50)
                guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=5.0)
                num_images_per_prompt = gr.Slider(minimum=1, maximum=10, step=1, label="Images per Prompt", value=1)
                btn = gr.Button("Generate Image")

            with gr.Column():
                output_images = gr.Gallery(label="Output Images", elem_id="output_gallery")

        btn.click(
            fn=infer,
            inputs=[prompt, use_random_seed, seed, height, width, num_inference_steps, guidance_scale, num_images_per_prompt],
            outputs=output_images
        )

    return demo

if __name__ == '__main__':
    gradio_interface().launch()
