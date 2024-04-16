# copied from https://github.com/huggingface/optimum/issues/1283
from pathlib import Path
from optimum.exporters import TasksManager
from optimum.exporters.onnx import onnx_export_from_model
from optimum.utils.save_utils import maybe_load_preprocessors
from transformers import Pix2StructForConditionalGeneration
import os

ORGANIZATION = "google"
MODEL_NAME = "pix2struct-screen2words-base"
MODELS_DIR = "model_repository"

MODEL_PATH = os.path.join(ORGANIZATION, MODEL_NAME)
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, 'pix2struct/1')

base_model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_PATH)
preprocessors = maybe_load_preprocessors(MODEL_PATH)

os.makedirs(ONNX_MODEL_PATH)
onnx_export_from_model(
                model=base_model,
                output=Path(ONNX_MODEL_PATH),
                monolith=True,
                do_validation=True,
                device='cpu',
                preprocessors=preprocessors,
                task='image-to-text',
            )