import torch
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from diffusers.utils import load_image
import os,sys

from kolors.pipelines.pipeline_controlnet_xl_kolors_img2img import StableDiffusionXLControlNetImg2ImgPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.models.controlnet import ControlNetModel

from diffusers import  AutoencoderKL
from kolors.models.unet_2d_condition import UNet2DConditionModel

from diffusers import EulerDiscreteScheduler
from PIL import Image
import numpy as np
import cv2

from annotator.midas import MidasDetector
from annotator.dwpose import DWposeDetector
from annotator.util import resize_image,HWC3

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def process_canny_condition( image, canny_threods=[100,200] ):
    np_image = image.copy()
    np_image = cv2.Canny(np_image, canny_threods[0], canny_threods[1])
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    np_image = HWC3(np_image)
    return Image.fromarray(np_image)


model_midas = None
def process_depth_condition_midas(img, res = 1024):
    h,w,_ = img.shape
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        model_midas = MidasDetector()
    
    result = HWC3( model_midas(img) )
    result = cv2.resize( result, (w,h) )
    return Image.fromarray(result)


model_dwpose = None
def process_dwpose_condition( image, res=1024 ):
    h,w,_ = image.shape
    img = resize_image(HWC3(image), res)
    global model_dwpose
    if model_dwpose is None:
        model_dwpose = DWposeDetector()
    out_res, out_img = model_dwpose(image) 
    result = HWC3( out_img )
    result = cv2.resize( result, (w,h) )
    return Image.fromarray(result)


def infer( image_path , prompt, model_type = 'Canny' ):

    ckpt_dir = f'{root_dir}/weights/Kolors'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

    control_path = f'{root_dir}/weights/Kolors-ControlNet-{model_type}'
    controlnet = ControlNetModel.from_pretrained( control_path , revision=None).half()

    pipe = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=vae,
            controlnet = controlnet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False
            )

    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    negative_prompt = 'nsfw，脸部阴影，低分辨率，jpeg伪影、模糊、糟糕，黑脸，霓虹灯'
    
    MAX_IMG_SIZE=1024
    controlnet_conditioning_scale = 0.7
    control_guidance_end = 0.9
    strength = 1.0

    basename = image_path.rsplit('/',1)[-1].rsplit('.',1)[0]

    init_image = Image.open( image_path )
    
    init_image = resize_image( init_image,  MAX_IMG_SIZE)
    if model_type == 'Canny':
        condi_img = process_canny_condition( np.array(init_image) )
    elif model_type == 'Depth':
        condi_img = process_depth_condition_midas( np.array(init_image), MAX_IMG_SIZE )
    elif model_type == 'Pose':
        condi_img = process_dwpose_condition( np.array(init_image), MAX_IMG_SIZE)

    generator = torch.Generator(device="cpu").manual_seed(66)
    image = pipe(
        prompt= prompt ,
        image = init_image,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        control_guidance_end = control_guidance_end, 
        strength= strength , 
        control_image = condi_img,
        negative_prompt= negative_prompt , 
        num_inference_steps= 50 , 
        guidance_scale= 6.0,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    
    condi_img.save( f'{root_dir}/controlnet/outputs/{model_type}_{basename}_condition.jpg' )
    image.save(f'{root_dir}/controlnet/outputs/{model_type}_{basename}.jpg')


if __name__ == '__main__':
    import fire
    fire.Fire(infer)
    

