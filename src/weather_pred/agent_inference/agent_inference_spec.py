"""
script: inferrence.py

This script defines the inferrence behavior for the deployable model.
"""

# Standard Python Library imports
import os
from pathlib import Path, PurePosixPath
import json
import joblib
import shutil

# Standard Library Type-Hinting

# Third-party Libraries
from sagemaker.serve.spec.inference_spec import InferenceSpec
import torch
import numpy as np

FEATURE_SCALER = None
LABEL_SCALER = None


class AgentInferenceSpec(InferenceSpec):
    """Class that defines the load and invoke functions that the builder will utilize for model handling"""

    def __init__(self, model_path=None):
        self.model_path = model_path

    def prepare(self, model_dir: str):
        """Prepare PyTorch model artifacts. Called once before container startup (prepararation phase)."""

        # print("Calling InferenceSpec prepare function ...", flush=True)
        print(
            "Calling InferenceSpec prepare function ... BUT RETURNING DUE TO CLOUD MODE", flush=True
        )

        pass

        # if not self.model_path:
        #     raise FileNotFoundError("'model_path' not set at initialization")
        # local_cont=False
        # if(local_cont):
        #     if os.path.isdir(self.model_path):
        #         # Copy directory contents to model_dir
        #         print(f"Copying to 'model_dir' - CONTENTS: {os.listdir(self.model_path)}", flush=True)
        #         for name in os.listdir(self.model_path):
        #             src = os.path.join(self.model_path, name)
        #             dst = os.path.join(model_dir, name)
        #             if os.path.isdir(src):
        #                 shutil.copytree(src, dst, dirs_exist_ok=True)
        #             else:
        #                 shutil.copy2(src, dst)
        #         print(f"Copied to '{model_dir}' - CONTENTS: {os.listdir(model_dir)}", flush=True)

        #         code_dir = os.path.join(model_dir, "code")
        #         os.makedirs(code_dir, exist_ok=True)

        #         # 2. Manually copy the src tree
        #         dest_src = os.path.join(code_dir, "src")
        #         if os.path.exists(dest_src):
        #             shutil.rmtree(dest_src)

        #         # Copying from your local WSL path (/mnt/c/Users/PROJECT_ROOT/src) to the staging model_dir
        #         shutil.copytree("src", dest_src)
        #         print(f"Copied to '{dest_src}' - CONTENTS: {os.listdir(dest_src)}", flush=True)

        # else:
        #     # Do nothing with a tarball
        #     print(
        #         f"Tarball case ({self.model_path}): ModelBuilder will unpack it to the container."
        #     )

    def load(self, model_dir: str = "/opt/ml/model"):
        """Loads a PyTorch model. Called inside the container when the model server initializes, usually with: model = spec.load(model_dir="/opt/ml/model")

        Args:
            model_dir (str): The path to the saved model.

        Returns:
            model (Agent): The loaded model.
        """
        print("CALLED AgentInferenceSpec LOAD FUNCTION", flush=True)
        print(f"Contents of {model_dir}: {os.listdir(model_dir)}")
        # Resolve dir when working on IN_PROCESS mode
        if not model_dir:
            model_dir = Path.home() / "local_mode/model"

        model_path = os.path.join(model_dir, "model.pt")

        #  --- Load Model ---
        if os.path.exists(model_path):
            model = torch.jit.load(model_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"TorchScript Model file does not exist: {model_path}")

        # --- Load joblib scalers once, cache globally for the runtime container ---
        global FEATURE_SCALER, LABEL_SCALER
        try:
            if FEATURE_SCALER is None:
                FEATURE_SCALER_PATH = os.environ.get(
                    "FEATURE_SCALER_URI", os.path.join(model_dir, "feature-scaler.joblib")
                )
                if FEATURE_SCALER_PATH:
                    print(f"Loading Feature Scaler from '{FEATURE_SCALER_PATH}'")
                    FEATURE_SCALER = joblib.load(FEATURE_SCALER_PATH)

            if LABEL_SCALER is None and os.environ.get(
                "LABEL_SCALER_URI"
            ):  # Only get if Label Scaler is needed by environment
                LABEL_SCALER_PATH = os.environ.get(
                    "LABEL_SCALER_URI", os.path.join(model_dir, "label-scaler.joblib")
                )  # optional if you inverse-transform labels
                if LABEL_SCALER_PATH:
                    LABEL_SCALER = joblib.load(LABEL_SCALER_PATH)
        except Exception as e:
            raise FileNotFoundError(f"Scaler file does not exist: {e}")

        print("Returning from the load function", flush=True)

        return model

    def invoke(self, payload, model):
        """Internal server logic that passes the payload's inputs through the model and returns the outputs. Called per real inference request. Usually called with: spec.invoke(input_obj, model)

        Args:
            payload: The HTTP payload.
            model (TorchScript): The trained model agent used for inference.

        Returns:
            (HTTP obj): The HTTP response.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Inside your load/invoke
        logger.info("--- ATTEMPTING INVOKE ---")
        print("CALLED AgentInferenceSpec INVOKE FUNCTION", flush=True)
        print(
            f"Payload content {payload} type: {type(payload)}",
        )
        try:
            if isinstance(payload, list):
                # Check if payload is already a dict (can happen during local build test)
                raw_inputs = payload
            elif isinstance(payload, str):
                raw_inputs = json.loads(payload)
            elif isinstance(payload, dict):
                raw_inputs = payload.get("inputs", payload.get("data"))
            else:
                # Standard cloud behavior: payload is bytes
                raw_inputs = json.loads(payload.decode("utf-8"))
                print("Payload needed to be deserialized", flush=True)

            print(f"raw_inputs: {raw_inputs}")

            scaled_inputs = self.preprocess(raw_inputs)
            print(f"scaled_inputs: {scaled_inputs}", flush=True)

            with torch.no_grad():
                outputs = model(scaled_inputs)

            body = json.dumps({"outputs": outputs.tolist()}).encode("utf-8")

            return body, "application/json", 200
        except json.JSONDecodeError as e:
            # Handle wrong encoding / binary response
            raise json.JSONDecodeError(
                e.encoding, e.object, e.start, e.end, f"Invalid decoding JSON: {e.reason}"
            )
        except KeyError as e:
            raise KeyError(f"{e}")
        except Exception as e:
            raise Exception(f"{e}")

    def preprocess(self, input_data: object):
        """Custom pre-processing function"""
        # Convert to numpy array for scaling
        input_data = np.asarray(input_data, dtype=np.float32)

        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)  # [1, n_features]

        scaled_input = FEATURE_SCALER.transform(input_data)

        return torch.tensor(scaled_input, dtype=torch.float32)


# This is the part SageMaker needs to find:
# inf_spec = AgentInferenceSpec()
