from optimum.onnxruntime import ORTModelForPix2Struct
from lavis.models import load_model_and_preprocess
from transformers import Pix2StructProcessor
from PIL import Image
from lavis.common.registry import registry
from fastapi import FastAPI, UploadFile
import torch
import os
import asyncio
import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ = torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ['INTRAOP_THREADS']))
torch.set_num_interop_threads(int(os.environ['INTEROP_THREADS']))

def get_blip_model_processor():
    # Use https://github.com/salesforce/LAVIS/issues/176#issuecomment-1459623682
    #model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)
    model_name = "blip_caption"
    model_type = "base_coco"
    model_class = registry.get_model_class(model_name)
    model_class.PRETRAINED_MODEL_CONFIG_DICT[model_type] = "/app/models/lavis-blip-base/blip_caption_base_coco.yaml"
    model, vis_processors, _ = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=DEVICE)
    return model, vis_processors["eval"]

def get_pix2struct_model_processor():
    model_id = "onnx/pix2struct_base"
    model = ORTModelForPix2Struct.from_pretrained(model_id)
    processor = Pix2StructProcessor.from_pretrained(model_id, is_vqa=False)
    return model, processor

BLIP_MODEL, BLIP_PROCESSOR = get_blip_model_processor()
P2S_MODEL, P2S_PROCESSOR = get_pix2struct_model_processor()

def blip_captions(images):
    inputs = [BLIP_PROCESSOR(image).unsqueeze(0).to(DEVICE) for image in images]
    inputs = torch.stack(inputs).squeeze(1).to(DEVICE)
    out = BLIP_MODEL.generate({"image": inputs}, num_beams=1, max_length=500)
    return [s.strip() for s in out]

def pix2struct_captions(images):
    inputs = P2S_PROCESSOR(images=images, return_tensors="pt").to(DEVICE)
    out = P2S_MODEL.generate(**inputs, max_new_tokens=500)
    return [s.strip() for s in P2S_PROCESSOR.batch_decode(out, skip_special_tokens=True)]

_ = blip_captions([Image.open("resources/000000001503.jpg")])
_ = pix2struct_captions([Image.open("resources/excel-spreadsheet.png")])

app = FastAPI()

@app.post("/caption/natural")
async def caption_natural_images(files: list[UploadFile]):
    contents = await asyncio.gather(*[file.read() for file in files])
    images = [Image.open(io.BytesIO(content)) for content in contents]
    return {'captions': blip_captions(images)}

@app.post("/caption/screenshot")
async def caption_screenshot_images(files: list[UploadFile]):
    contents = await asyncio.gather(*[file.read() for file in files])
    images = [Image.open(io.BytesIO(content)) for content in contents]
    b_cap = blip_captions(images)
    p2s_cap = pix2struct_captions(images)
    captions = [cap1 + "; " + cap2 for cap1, cap2 in zip(p2s_cap, b_cap)]
    return {'captions': captions}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ['PORT'], debug=True)