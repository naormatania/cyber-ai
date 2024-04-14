from lavis.models import load_model_and_preprocess
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)