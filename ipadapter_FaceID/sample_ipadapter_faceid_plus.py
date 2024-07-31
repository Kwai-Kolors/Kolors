import torch
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from diffusers.utils import load_image
import os,sys

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter_FaceID import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer

from diffusers import  AutoencoderKL
from kolors.models.unet_2d_condition import UNet2DConditionModel

from diffusers import EulerDiscreteScheduler
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import insightface
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

class FaceInfoGenerator():
    def __init__(self, root_dir = "./"):
        self.app = FaceAnalysis(name = 'antelopev2', root = root_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id = 0, det_size = (640, 640))

    def get_faceinfo_one_img(self, image_path):
        face_image = load_image(image_path)
        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))

        if len(face_info) == 0:
            face_info = None
        else:
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
        return face_info

def face_bbox_to_square(bbox):
    ## l, t, r, b to square l, t, r, b
    l,t,r,b = bbox
    cent_x = (l + r) / 2
    cent_y = (t + b) / 2
    w, h = r - l, b - t
    r = max(w, h) / 2

    l0 = cent_x - r
    r0 = cent_x + r
    t0 = cent_y - r
    b0 = cent_y + r

    return [l0, t0, r0, b0]

def infer(test_image_path, text_prompt):
    ckpt_dir = f'{root_dir}/weights/Kolors'
    ip_model_dir = f'{root_dir}/weights/Kolors-IP-Adapter-FaceID-Plus'
    device = "cuda:0"

    #### base Kolors model
    text_encoder = ChatGLMModel.from_pretrained( f'{ckpt_dir}/text_encoder', torch_dtype = torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f'{ckpt_dir}/vae', subfolder = "vae", revision = None)
    scheduler = EulerDiscreteScheduler.from_pretrained(f'{ckpt_dir}/scheduler')
    unet = UNet2DConditionModel.from_pretrained(f'{ckpt_dir}/unet', revision = None).half()

    #### clip image encoder for face structure
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(f'{ip_model_dir}/clip-vit-large-patch14-336', ignore_mismatched_sizes=True)
    clip_image_encoder.to(device)
    clip_image_processor = CLIPImageProcessor(size = 336, crop_size = 336)

    pipe = StableDiffusionXLPipeline(
            vae = vae,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            unet = unet,
            scheduler = scheduler,
            face_clip_encoder = clip_image_encoder,
            face_clip_processor = clip_image_processor,
            force_zeros_for_empty_prompt = False,
        )
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()

    pipe.load_ip_adapter_faceid_plus(f'{ip_model_dir}/ipa-faceid-plus.bin', device = device)

    scale = 0.8
    pipe.set_face_fidelity_scale(scale)

    #### prepare face embedding & bbox with insightface toolbox
    face_info_generator = FaceInfoGenerator(root_dir = "./")
    img = Image.open(test_image_path)
    face_info = face_info_generator.get_faceinfo_one_img(test_image_path)

    face_bbox_square = face_bbox_to_square(face_info["bbox"])
    crop_image = img.crop(face_bbox_square)
    crop_image = crop_image.resize((336, 336))
    crop_image = [crop_image]

    face_embeds = torch.from_numpy(np.array([face_info["embedding"]]))
    face_embeds = face_embeds.to(device, dtype = torch.float16)

    #### generate image
    generator = torch.Generator(device = device).manual_seed(66)
    image = pipe(
        prompt = text_prompt,
        negative_prompt = "", 
        height = 1024,
        width = 1024,
        num_inference_steps= 25, 
        guidance_scale = 5.0,
        num_images_per_prompt = 1,
        generator = generator,
        face_crop_image = crop_image,
        face_insightface_embeds = face_embeds,
    ).images[0]
    image.save(f'../scripts/outputs/test_res.png')

if __name__ == '__main__':
    import fire
    fire.Fire(infer)