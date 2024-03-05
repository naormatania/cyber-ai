from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoProcessor
from lavis.models import load_model_and_preprocess
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

BLIP_MODELS = {
    "base": "models/blip-base/",
    "large": "models/blip-large/",
}
BLIP_MODEL_ID = BLIP_MODELS["large"]
BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
BLIP_MODEL = BLIP_MODEL.to(DEVICE)
BLIP_PROCESSOR = BlipProcessor.from_pretrained(BLIP_MODEL_ID)

GIT_MODELS = {
    "base": "models/git-base/",
    "large": "models/git-large/",
    "large-textcaps": "models/git-large-textcaps/",
}
GIT_MODEL_ID = GIT_MODELS["large"]
GIT_MODEL = AutoModelForCausalLM.from_pretrained(GIT_MODEL_ID)
GIT_MODEL = GIT_MODEL.to(DEVICE)
GIT_PROCESSOR = AutoProcessor.from_pretrained(GIT_MODEL_ID)

# TODO: The docker fails to load the model due to OOM. Fix it
# BLIP2_MODEL_ID = "models/blip2/"
# BLIP2_MODEL = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL_ID)
# BLIP2_MODEL = BLIP2_MODEL.to(DEVICE)
# BLIP2_PROCESSOR = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)

# LBLIP2_MODEL_BASE, LBLIP2_PROCESSORS_BASE, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)
LBLIP2_MODEL_LARGE, LBLIP2_PROCESSORS_LARGE, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=DEVICE)
LBLIP2_PROCESSOR = LBLIP2_PROCESSORS_LARGE["eval"]
LBLIP2_MODEL = LBLIP2_MODEL_LARGE

# ONNX is not supported
optimization = os.environ['OPTIMIZATION']
if optimization == "TORCH_COMPILE": 
    print("torch compile")
    BLIP_MODEL = torch.compile(BLIP_MODEL)
    GIT_MODEL = torch.compile(GIT_MODEL)
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

@app.post("/caption/blip2-lavis")
async def caption_images_blip2_lvais(files: list[UploadFile], min_length: int | None = None):
    contents = await asyncio.gather(*[file.read() for file in files])
    images = [Image.open(io.BytesIO(content)).convert('RGB') for content in contents]

    captions = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        inputs = [LBLIP2_PROCESSOR(raw_image).unsqueeze(0).to(DEVICE) for raw_image in batch]
        inputs = torch.stack(inputs).squeeze(1).to(DEVICE)
        if min_length is not None:
            out = LBLIP2_MODEL.generate({"image": inputs}, num_beams=1, max_length=500, min_length=min_length)
        else:
            out = LBLIP2_MODEL.generate({"image": inputs}, num_beams=1, max_length=500)
        captions.extend(out)
    
    return {'captions': [captions]}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)