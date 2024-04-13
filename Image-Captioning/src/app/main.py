from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, Pix2StructForConditionalGeneration, AutoProcessor
from lavis.models import load_model_and_preprocess
from optimum.onnxruntime import ORTModelForPix2Struct, ORTModelForVision2Seq
from fastapi import FastAPI, UploadFile
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from optimum.bettertransformer import BetterTransformer
from PIL import Image
import asyncio
import io
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ['BATCH_SIZE'])

_ = torch.set_grad_enabled(False)
print("default num threads for intraop parallelism: ", torch.get_num_threads())
torch.set_num_threads(int(os.environ['INTRAOP_THREADS'])) # intraop parallelism on CPU
print("current num threads for intraop parallelism: ", torch.get_num_threads())

print("default num threads for interop parallelism: ", torch.get_num_interop_threads())
torch.set_num_interop_threads(int(os.environ['INTEROP_THREADS'])) # interop parallelism on CPU
print("current num threads for interop parallelism: ", torch.get_num_interop_threads())

# BLIP_MODELS = {
#     "base": "models/blip-base/",
#     "large": "models/blip-large/",
# }
# BLIP_MODEL_ID = BLIP_MODELS["large"]
# BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
# BLIP_PROCESSOR = BlipProcessor.from_pretrained(BLIP_MODEL_ID)

GIT_MODELS = {
    "base": "models/git-base/",
    "large": "models/git-large/",
    "large-textcaps": "models/git-large-textcaps/",
    "base-coco": "models/git-base-coco/",
    "large-coco": "models/git-large-coco/",
}
# GIT_MODEL_ID = GIT_MODELS["base"]
# GIT_MODEL = AutoModelForCausalLM.from_pretrained(GIT_MODEL_ID)
# GIT_PROCESSOR = AutoProcessor.from_pretrained(GIT_MODEL_ID)
# GIT_MODEL_ID = GIT_MODELS["large-coco"]
# GIT_MODEL = AutoModelForCausalLM.from_pretrained(GIT_MODEL_ID)
# GIT_PROCESSOR = AutoProcessor.from_pretrained(GIT_MODEL_ID)

PIX2STRUCT_MODELS = {
    "base": "models/pix2struct-base/",
    "large": "models/pix2struct-large/",
    "base-onnx": "onnx/pix2struct_base/"
}
# PIX2STRUCT_BASE_MODEL = Pix2StructForConditionalGeneration.from_pretrained(PIX2STRUCT_MODELS["base"])
PIX2STRUCT_BASE_MODEL = ORTModelForPix2Struct.from_pretrained(PIX2STRUCT_MODELS["base-onnx"])
PIX2STRUCT_BASE_PROCESSOR = AutoProcessor.from_pretrained(PIX2STRUCT_MODELS["base"])

# PIX2STRUCT_LARGE_MODEL = Pix2StructForConditionalGeneration.from_pretrained(PIX2STRUCT_MODELS["large"])
# PIX2STRUCT_LARGE_PROCESSOR = AutoProcessor.from_pretrained(PIX2STRUCT_MODELS["large"])

# TODO: The docker fails to load the model due to OOM. Fix it
# BLIP2_MODEL_ID = "models/blip2/"
# BLIP2_MODEL = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL_ID)
# BLIP2_PROCESSOR = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)

# LBLIP_MODEL_BASE, LBLIP_PROCESSORS_BASE, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)
# LBLIP_MODEL_LARGE, LBLIP_PROCESSORS_LARGE, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=DEVICE)
# LBLIP_PROCESSOR = LBLIP_PROCESSORS_LARGE["eval"]
# LBLIP_MODEL = LBLIP_MODEL_LARGE


# ONNX is not supported
optimization = os.environ['OPTIMIZATION']
if optimization == "TORCH_COMPILE":
    print("torch compile")
    # BLIP_MODEL = torch.compile(BLIP_MODEL)
    # GIT_MODEL = torch.compile(GIT_MODEL)
    # PIX2STRUCT_BASE_MODEL = torch.compile(PIX2STRUCT_BASE_MODEL)
    # PIX2STRUCT_LARGE_MODEL = torch.compile(PIX2STRUCT_LARGE_MODEL)
    # This does not improve LBLIP2
    # LBLIP2_MODEL = torch.compile(LBLIP2_MODEL)
    #BLIP2_MODEL = torch.compile(BLIP2_MODEL)
# elif optimization == "BETTER_TRANSFORMER":
#     BLIP2_MODEL = BetterTransformer.transform(BLIP2_MODEL)

app = FastAPI()

def caption(model, preprocessor, images, min_new_tokens):
    captions = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        inputs = preprocessor(images=batch, return_tensors="pt").to(DEVICE)
        out = model.generate(**inputs, max_new_tokens=500, min_new_tokens=min_new_tokens)#, num_beams=5, early_stopping=True, num_return_sequences=5)
        captions.extend(preprocessor.batch_decode(out, skip_special_tokens=True))
    return captions

@app.post("/caption/blip")
async def caption_images_blip(files: list[UploadFile], min_new_tokens: int | None = None):
    contents = await asyncio.gather(*[file.read() for file in files])
    images = [Image.open(io.BytesIO(content)) for content in contents]
    captions = caption(BLIP_MODEL, BLIP_PROCESSOR, images, min_new_tokens)
    return {'captions': [captions]}

@app.post("/caption/git")
async def caption_images_git(files: list[UploadFile], min_new_tokens: int | None = None):
    contents = await asyncio.gather(*[file.read() for file in files])
    images = [Image.open(io.BytesIO(content)) for content in contents]
    captions = caption(GIT_MODEL, GIT_PROCESSOR, images, min_new_tokens)
    return {'captions': [captions]}

# @app.post("/caption/blip2")
# async def caption_images_blip2(files: list[UploadFile], min_new_tokens: int | None = None):
#     contents = await asyncio.gather(*[file.read() for file in files])
#     images = [Image.open(io.BytesIO(content)) for content in contents]
#     captions = caption(BLIP2_MODEL, BLIP2_PROCESSOR, images, min_new_tokens)
#     return {'captions': [captions]}

@app.post("/caption/blip-lavis")
async def caption_images_blip_lavis(files: list[UploadFile], min_length: int | None = None):
    contents = await asyncio.gather(*[file.read() for file in files])
    images = [Image.open(io.BytesIO(content)).convert('RGB') for content in contents]

    captions = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        inputs = [LBLIP_PROCESSOR(raw_image).unsqueeze(0).to(DEVICE) for raw_image in batch]
        inputs = torch.stack(inputs).squeeze(1).to(DEVICE)
        if min_length is not None:
            out = LBLIP_MODEL.generate({"image": inputs}, num_beams=1, max_length=500, min_length=min_length)
        else:
            out = LBLIP_MODEL.generate({"image": inputs}, num_beams=1, max_length=500)
        captions.extend(out)
    
    return {'captions': [captions]}

@app.post("/caption/pix2struct-base")
async def caption_images_pix2struct_base(files: list[UploadFile], min_new_tokens: int | None = None):
    contents = await asyncio.gather(*[file.read() for file in files])
    images = [Image.open(io.BytesIO(content)) for content in contents]
    captions = caption(PIX2STRUCT_BASE_MODEL, PIX2STRUCT_BASE_PROCESSOR, images, min_new_tokens)
    return {'captions': [captions]}

@app.post("/caption/pix2struct-large")
async def caption_images_pix2struct_large(files: list[UploadFile], min_new_tokens: int | None = None):
    contents = await asyncio.gather(*[file.read() for file in files])
    images = [Image.open(io.BytesIO(content)) for content in contents]
    captions = caption(PIX2STRUCT_LARGE_MODEL, PIX2STRUCT_LARGE_PROCESSOR, images, min_new_tokens)
    return {'captions': [captions]}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)