import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image

_ = torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

class TritonPythonModel:
    def initialize(self, args):
        model_id = "google/pix2struct-screen2words-base"
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
        self.processor = Pix2StructProcessor.from_pretrained(model_id, is_vqa=False)
    
    def execute(self, requests):
        responses = []
        for request in requests:
            # Decode the Byte Tensor into Text 
            image = pb_utils.get_input_tensor_by_name(request, "image")
            image = Image.fromarray(image.as_numpy())

            inputs = self.processor(images=[image], return_tensors="pt")
            out = self.model.generate(**inputs, max_new_tokens=500)
            output = [s.strip() for s in self.processor.batch_decode(out, skip_special_tokens=True)]
                        
            # Encode the text to byte tensor to send back
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "caption",
                        np.array([o.encode() for o in output]),
                        )
                ]
            )
            responses.append(inference_response)
        
        return responses
