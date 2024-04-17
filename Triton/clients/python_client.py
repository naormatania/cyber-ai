import tritonclient.http as httpclient
from PIL import Image
import numpy as np

triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000', concurrency=1)

image = Image.open("../Image-Captioning/datasets/desktop-ui-dataset/images/access-data-source-admin.png")
image = np.array(image)

inputs = [httpclient.InferInput("image", image.shape, 'UINT8')]
inputs[0].set_data_from_numpy(image)
outputs = [httpclient.InferRequestedOutput('caption')]

res = triton_client.infer('pix2struct', inputs, request_id="1", model_version="1", outputs=outputs)
print(res.as_numpy('caption'))
