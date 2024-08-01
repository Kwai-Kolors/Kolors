import os, torch
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

from peft import (
    LoraConfig,
    PeftModel,
)

def infer(prompt):
    ckpt_dir = "/path/base_model_path"
    lora_ckpt = 'trained_models/ktxl_dog_text/checkpoint-1000/'
    load_text_encoder = True

    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()
    pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False)
    pipe = pipe.to("cuda")

    pipe.load_lora_weights(lora_ckpt, adapter_name="ktxl-lora") 
    pipe.set_adapters(["ktxl-lora"], [0.8])

    if load_text_encoder:
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, lora_ckpt)

    random_seed = 0
    generator = torch.Generator(pipe.device).manual_seed(random_seed)

    neg_p = ''
    out = pipe(prompt, generator=generator, negative_prompt=neg_p, num_inference_steps=25, width=1024, height=1024, num_images_per_prompt=1, guidance_scale=5).images[0]
    out.save("ktxl_test_image.png")

if __name__ == '__main__':
    import fire
    fire.Fire(infer)