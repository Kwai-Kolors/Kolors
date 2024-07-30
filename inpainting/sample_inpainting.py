import torch
import os, sys
from PIL import Image

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import StableDiffusionXLInpaintPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer

def infer(image_path, mask_path, prompt):

    ckpt_dir = f'{root_dir}/weights/Kolors-Inpainting'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

    pipe = StableDiffusionXLInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
    )
    
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(603)
    basename = image_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    image = Image.open(image_path).convert('RGB')
    mask_image = Image.open(mask_path).convert('RGB')

    result = pipe(
        prompt = prompt,
        image = image,
        mask_image = mask_image,
        height=1024,
        width=768,
        guidance_scale = 6.0,
        generator= generator,
        num_inference_steps= 25,
        negative_prompt = '残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量',
        num_images_per_prompt = 1,
        strength = 0.999
    ).images[0]
    result.save(f'{root_dir}/scripts/outputs/sample_inpainting_{basename}.jpg')

if __name__ == '__main__':
    import fire
    fire.Fire(infer)
