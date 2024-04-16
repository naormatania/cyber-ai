## Triton
```
python3 export_pix2struct_to_single_onnx.py
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/Users/naormatania/code/cyber-ai/Triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.03-py3 tritonserver --model-repository=/models
pip install tritonclient[all]
```
Relevant example for usage: https://github.com/triton-inference-server/client/blob/main/src/python/examples/image_client.py
```
!pip install tritonclient\[all\] transformers pillow torch
import tritonclient.http as httpclient
from transformers import Pix2StructProcessor
from PIL import Image
import numpy as np
import torch.nn.functional as F

triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000', concurrency=1)
model_metadata = triton_client.get_model_metadata(model_name='pix2struct', model_version='1')
model_config = triton_client.get_model_config(model_name='pix2struct', model_version='1')
inputs_metadata = model_metadata["inputs"]
outputs_metadata = model_metadata["outputs"]

inputs[0].set_data_from_numpy(batched_image_data)
outputs = [client.InferRequestedOutput(output_name, class_count=FLAGS.classes)]

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
``` 