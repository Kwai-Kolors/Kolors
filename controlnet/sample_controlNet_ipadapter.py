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


def infer( image_path , ip_image_path,  prompt, model_type = 'Canny' ):

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

    # IP-Adapter model
    image_encoder = CLIPVisionModelWithProjection.from_pretrained( f'{root_dir}/weights/Kolors-IP-Adapter-Plus/image_encoder',  ignore_mismatched_sizes=True).to(dtype=torch.float16)
    ip_img_size = 336
    clip_image_processor = CLIPImageProcessor( size=ip_img_size, crop_size=ip_img_size )
    
    pipe = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=vae,
            controlnet = controlnet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=clip_image_processor,
            force_zeros_for_empty_prompt=False
            )

    if hasattr(pipe.unet, 'encoder_hid_proj'):
        pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj
    
    pipe.load_ip_adapter( f'{root_dir}/weights/Kolors-IP-Adapter-Plus' , subfolder="", weight_name=["ip_adapter_plus_general.bin"])

    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    negative_prompt = 'nsfw，脸部阴影，低分辨率，糟糕的解剖结构、糟糕的手，缺失手指、质量最差、低质量、jpeg伪影、模糊、糟糕，黑脸，霓虹灯'
    
    MAX_IMG_SIZE=1024
    controlnet_conditioning_scale = 0.5
    control_guidance_end = 0.9
    strength = 1.0
    ip_scale = 0.5
    
    basename = image_path.rsplit('/',1)[-1].rsplit('.',1)[0]

    init_image = Image.open( image_path )
    init_image = resize_image( init_image,  MAX_IMG_SIZE)
    
    if model_type == 'Canny':
        condi_img = process_canny_condition( np.array(init_image) )
    elif model_type == 'Depth':
        condi_img = process_depth_condition_midas( np.array(init_image), MAX_IMG_SIZE )

    ip_adapter_img = Image.open(ip_image_path)
    pipe.set_ip_adapter_scale([ ip_scale ])
    
    generator = torch.Generator(device="cpu").manual_seed(66)
    image = pipe(
        prompt= prompt ,
        image = init_image,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        control_guidance_end = control_guidance_end, 

        ip_adapter_image=[ ip_adapter_img ],
        
        strength= strength , 
        control_image = condi_img,
        negative_prompt= negative_prompt , 
        num_inference_steps= 50 , 
        guidance_scale= 5.0,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    
    image.save(f'{root_dir}/controlnet/outputs/{model_type}_ipadapter_{basename}.jpg')
    condi_img.save(f'{root_dir}/controlnet/outputs/{model_type}_{basename}_condition.jpg')


if __name__ == '__main__':
    import fire
    fire.Fire(infer)
    









