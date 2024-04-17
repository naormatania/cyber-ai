import tritonclient.http as httpclient
from transformers import Pix2StructProcessor
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch

triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000', concurrency=1)
model_metadata = triton_client.get_model_metadata(model_name='pix2struct', model_version='1')
model_config = triton_client.get_model_config(model_name='pix2struct', model_version='1')
inputs_metadata = model_metadata["inputs"]
outputs_metadata = model_metadata["outputs"]

images = [Image.open("../Image-Captioning/datasets/desktop-ui-dataset/images/access-data-source-admin.png")]
processor = Pix2StructProcessor.from_pretrained('google/pix2struct-screen2words-base', is_vqa=False)
processed_inputs = processor(images=images, return_tensors="pt")

inputs = [httpclient.InferInput("flattened_patches", processed_inputs['flattened_patches'].shape, 'FP32'), httpclient.InferInput("attention_mask", processed_inputs['attention_mask'].shape, 'INT64'), httpclient.InferInput("decoder_input_ids", torch.tensor([[1]]).shape, 'INT64')]

inputs[0].set_data_from_numpy(np.array(processed_inputs['flattened_patches']))
inputs[1].set_data_from_numpy(np.array(processed_inputs['attention_mask']).astype(int))
inputs[2].set_data_from_numpy(np.array([[0]]).astype(int))

outputs = [httpclient.InferRequestedOutput('logits')]

res = triton_client.infer('pix2struct', inputs, request_id="1", model_version="1", outputs=outputs)
preds = F.softmax(torch.tensor(res.as_numpy("logits")), dim=-1).argmax(dim=-1)
processor.batch_decode(preds, skip_special_tokens=True)