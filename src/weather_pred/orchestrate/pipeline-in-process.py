"""
Script: pipeline.py

Description: Testing out Sagemaker training and inferrence locally using IN_PROCESS MODE.
"""

# Standard Python Libraries
import os, pathlib
from pathlib import Path, PurePosixPath
import logging
import shutil
from datetime import datetime
import json
import tarfile
from urllib.parse import urlparse

# Third-Party Libraries
import torch
from sagemaker.train import ModelTrainer
from sagemaker.serve import ModelBuilder
from sagemaker.train.model_trainer import ModelTrainer, Mode as TrainingMode
from sagemaker.serve.mode.function_pointers import Mode as ServeMode
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.configs import SourceCode, InputData, OutputDataConfig, Compute
import boto3
from botocore.config import Config

# Third-Party Exceptions
from sagemaker.core.utils.exceptions import ValidationError

# User-Defined Libraries
from weather_pred import get_uri, AgentInferenceSpec


RUN_LOCAL = True  # Toggle this to False for managed SageMaker Operations

SOURCE_DIR = (
    Path(__file__).resolve().parents[2]
)  # Go up 2 levels: orchestrate/ → package_name → src
# Set-up a location on the Host to mount to; Used to share/store files
CONTAINER_ROOT = Path.home() / "local_mode"  # /home/<user_name>/local_mode/
CONTAINER_ROOT.mkdir(parents=True, exist_ok=True)
# AWS S3 Bucket Name
AWS_BUCKET = "alexandria-reborn"

# # AWS Variables
AWS_REGION = "us-east-1"

PROJECT_PREFIX = "CA-Weather-Fire"
DATA_SPLITS_DIR = "DataSplits"
# scaler_dir = "Scalers"

PROFILE_NAME = "default"  # Must be defined in the "~/.aws/credentials" file
# Generate a unique string for each
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
JOB_NAME = f"weather-trainer-{timestamp}"
MODEL_NAME = f"weather-model-{timestamp}"
ENDPOINT_NAME = f"weather-endpoint-{timestamp}"

# Create Boto3 Session with profile and copy credentials
boto3_session = boto3.Session(profile_name=PROFILE_NAME, region_name=AWS_REGION)
# 2) Region-bound SageMaker client (explicit)
sm_client = boto3_session.client(service_name="sagemaker", region_name=AWS_REGION)
s3_client = boto3_session.client(service_name="s3", region_name=AWS_REGION)
sm_sess = Session(
    boto_session=boto3_session,
    sagemaker_client=sm_client,
    default_bucket=AWS_BUCKET,
    default_bucket_prefix=PROJECT_PREFIX,
)

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION,
    config=Config(read_timeout=300, retries={"max_attempts": 3}),
)

SAGEMAKER_ROLE = (
    "arn:aws:iam::532494224167:role/service-role/AmazonSageMaker-ExecutionRole-20251215T115387"
)

logger = logging.getLogger(__name__)  # Create global logger

# Bucket-Relative Path Setup
DATA_SPLITS_DIR = PurePosixPath(PROJECT_PREFIX) / DATA_SPLITS_DIR

TRAIN_DATA_PATH = DATA_SPLITS_DIR / "train.csv"

# # AWS Paths
TRAIN_DATA_PATH = get_uri(logger, AWS_BUCKET, TRAIN_DATA_PATH)

FINAL_TARBALL_OUTPUT_S3_PATH = "s3://alexandria-reborn/CA-Weather-Fire/Compressed_Artifacts/"  # Where ModelBuilder will upload the packaged model

# Configure AWS EC2 Compute Images
TRAIN_IMAGE_URI = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker"
INFER_IMAGE_URI = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker"
"763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker"

# Instance types and extra arguments
TRAIN_INSTANCE_TYPE = "ml.c5.xlarge"
TRAIN_INSTANCE_COUNT = 1
TRAIN_VOLUME_SIZE_GB = 15
TRAIN_MODE = TrainingMode.LOCAL_CONTAINER

WEIGHTS_FILE_NAME = "model.pt"
LOCAL_TAR = "model.tar.gz"
LOCAL_BUILD_DIR = "./DEPLOYMENT_VALIDATION"
INFER_INSTANCE_TYPE = "ml.c5.xlarge"
# prefix with 'ml.'
# c6g.medium - 1 vCPU | 2 GiB
# c6g.large - 2 vCPU | 4 GiB
# c5.xlarge - 4 vCPU | 8 GiB

SERVE_MODE = ServeMode.IN_PROCESS  # Local FastAPI server (no Docker, no TorchServe).


SCHEMA_SAMPLE_INPUTS = [[312.0, 0.17, 1.06, 8.05, 52.0]]
SCHEMA_SAMPLE_OUTPUTS = [[62.0]]


# --- Environment variables passed into the serving container ---
env_vars = {
    "AWS_REGION": AWS_REGION,
    "AWS_BUCKET": AWS_BUCKET,
    "TRAIN_DATA_S3_URI": TRAIN_DATA_PATH,
    # "AWS_PROFILE": profile_name,  # Causes errors without the volume mounting
    "TRAINING_JOB_NAME": "local-trainer",
    "PYTHONPATH": "/opt/ml/input/data/code",  # Resolves the importing config.json file issue
    "SAVE_MODEL": "true",  # Has to be 'true' for pipeline to work
    "MODEL_WEIGHTS_FILE_NAME": WEIGHTS_FILE_NAME,  # Use .pt or .pth, just NOT 'model.pt'
    # For S3 scalers (bucket-relative keys)
    "FEATURE_SCALER_URI": "CA-Weather-Fire/Scalers/feature-scaler.joblib",  # Bucket-relative!!
    "LABEL_SCALER_URI": "",
    # "CUSTOM_MODEL_PATH": "/opt/ml/model/" + WEIGHTS_FILE_NAME,
}


# If local & reading scalers from S3, include temporary creds & region.
if RUN_LOCAL:
    # Only necessary if your container fetches from S3 in local mode

    creds = boto3_session.get_credentials().get_frozen_credentials()

    env_vars.update(
        {
            "AWS_ACCESS_KEY_ID": creds.access_key,
            "AWS_SECRET_ACCESS_KEY": creds.secret_key,
        }
    )
    if creds.token:
        env_vars["AWS_SESSION_TOKEN"] = creds.token

    TMPDIR = "/home/congajamm/local_mode/tmp"
    pathlib.Path(TMPDIR).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = TMPDIR


custom_dependencies = []


def main():

    src = SourceCode(
        source_dir=str(SOURCE_DIR),  # Use the "src" directory
        # Everything in the `source_dir` will be copied to `/opt/ml/input/data/code/**`
        entry_script="weather_pred/training/train.py",  # Relative to the source_dir
        # command="python train.py epochs=1",
        ignore_patterns=[
            ".env",
            ".git",
            "__pycache__",
            ".DS_Store",
            ".cache",
            ".ipynb_checkpoints",
        ],
    )
    compute = Compute(
        instance_type=TRAIN_INSTANCE_TYPE,
        instance_count=TRAIN_INSTANCE_COUNT,
        volume_size_in_gb=TRAIN_VOLUME_SIZE_GB,
    )

    trainer = ModelTrainer(
        role=SAGEMAKER_ROLE,  # Not used in local mode, but param is required
        # Force a known local mount root for artifacts
        sagemaker_session=sm_sess,
        training_mode=TRAIN_MODE,
        base_job_name="local-trainer",
        source_code=src,
        training_image=TRAIN_IMAGE_URI,
        training_input_mode="File",  # or Pipe, FastFile
        environment=env_vars,
        local_container_root=str(
            CONTAINER_ROOT
        ),  # Ensures transfers everything from /opt/ml/model -> local_container_root
    )  # Moves "model" and "outputs" to "artifacts" dir, then compresses to "compressed_artifacts"

    # Using CLOUD DATA
    trainer.train(
        logs=True
    )  # Starts training on local Docker container without any data download; copies the code, and runs the source_code
    # logger.info(f"Check out new content in '{CONTAINER_ROOT}'. Includes the 'model', 'input', 'outputs' and other directories")
    if RUN_LOCAL:
        # Define paths
        model_artifacts_locator = str(
            CONTAINER_ROOT / "artifacts" / "model"
        )  # Path on your WSL drive
        # model_artifacts_locator = str(CONTAINER_ROOT / "compressed_artifacts" / "model.tar.gz")  # Path on your WSL drive

        trained_weights = CONTAINER_ROOT / "model" / WEIGHTS_FILE_NAME
        inference_staging = SOURCE_DIR / "weather_pred" / WEIGHTS_FILE_NAME

    else:
        # Get Trainer Output Path in S3
        # job_name = trainer._latest_training_job.training_job_name

        # desc = sm_client.describe_training_job(TrainingJobName=job_name)

        # model_artifacts_locator = desc["ModelArtifacts"]["S3ModelArtifacts"]

        model_artifacts_locator = "s3://alexandria-reborn/CA-Weather-Fire/local-trainer/local-trainer-20260113113310/output/model.tar.gz"

        parsed_uri = urlparse(model_artifacts_locator)
        bucket = parsed_uri.netloc
        key = parsed_uri.path.lstrip("/")

        print(f"Downloading artifacts from \n{model_artifacts_locator} to {LOCAL_TAR}")
        s3_client.download_file(Bucket=bucket, Key=key, Filename=LOCAL_TAR)

        os.makedirs(LOCAL_BUILD_DIR, exist_ok=True)
        try:
            with tarfile.open(LOCAL_TAR) as tar:
                tar.extractall(LOCAL_BUILD_DIR)
        except Exception as e:
            logger.error()
        logger.info(
            f"\n=====================================\
                    \nBUILDING TRAINED MODEL from '{LOCAL_BUILD_DIR}'"
        )
    # Instantiate a ModelBuilder Orchestrator "Wrapper"
    # In managed mode, when you pass a tarball to ModelBuilder.model_path, the SDK unpacks it into the container’s /opt/ml/model (or equivalent staging dir) for you. You don’t need to manually copy individual files in inference_spec.prepare()

    # Uses TorchServe + the SageMaker PyTorch Inference Toolkit to run everything (decoder → default_input_fn → predict_fn → output_fn)
    builder = ModelBuilder(
        mode=SERVE_MODE,
        model_path=model_artifacts_locator,  # (directory or .tar.gz)
        role_arn=SAGEMAKER_ROLE,
        env_vars={
            **env_vars,
            "MODEL_ARTIFACT_S3": model_artifacts_locator,
            # "SAGEMAKER_PROGRAM": "inference.py"
        },
        log_level=logging.DEBUG,
        schema_builder=SchemaBuilder(
            sample_input=SCHEMA_SAMPLE_INPUTS, sample_output=SCHEMA_SAMPLE_OUTPUTS
        ),
        inference_spec=AgentInferenceSpec(),
    )

    # Take the model_path (directory or .tar.gz)
    # Produce a model package tailored to the chosen model server (e.g., TorchServe) within model_path/code (serve.pkl)
    # Copy your artifacts into the serving image’s '/opt/ml/model/' at runtime
    # Builds serving artifacts within container returning a deployable model object (not an endpoint)
    # Packages code + dependencies per the server Returning a deployable Model object (configuration + artifact metadata)
    # Uploads the model package to S3 if provided in ModelBuilder Instantiation
    built_model = builder.build(
        model_name=MODEL_NAME,
        mode=SERVE_MODE,
        # role_arn="defalult-role",
        sagemaker_session=sm_sess,
    )

    endpoint = None  # Initialize to ensure finally block can check it
    try:
        if RUN_LOCAL:
            logger.info("Starting local container deployment...")
        else:
            logger.info("Starting sagemaker MANAGED container deployment...")
        logger.info("Note: This may take a few minutes to pull the Docker image on first run.")

        # ModelBuilder.deploy() returns a runtime Endpoint handle created with the model artifacts created by build().
        # Provide instance type, count, serverless settings here.
        # The model must be built before calling deploy()
        # Starts inference endpoint on Docker container and returns a predictor/endpoint locally
        endpoint = builder.deploy_local(
            endpoint_name=ENDPOINT_NAME,
            wait=True,  # The call should wait until the deployment completes.
            update_endpoint=False,  # Update an existing endpoint in-place vs create new.
            container_timeout_in_seconds=300,  # Max time to wait for container to become healthy
        )

        # Sample the model ;
        inputs = [[345.1, 0.18, 1.07, 9.05, 51.0]]
        payload = json.dumps(inputs)
        try:
            # Sagemaker is able to dictate what to do to serialize the payload based on the SchemaBuilder
            response = endpoint.invoke(
                body=payload,
                content_type="application/json",  # MIME type of the input payload data
                accept="application/json",  # Desired MIME type  response from the model container
            )  # This is the "external" trigger to the AWS Endpoint; expects raw bytes
            # logger.info(f"Returned from endpoint.invoke, type={type(response)}")

        except ValidationError as e:
            raise ValidationError(f"Validation error during 'invoke': {e}")
        except Exception as e:
            raise Exception(f"Error during invoke function call: {e}")
        # If 'response' is a dictionary (common in SageMaker Core/Boto3 style)
        if isinstance(response, dict) and "Body" in response:
            logger.info("Response is a dictionary and contains a 'Body' in response")
            model_output = json.loads(response["Body"].read().decode("utf-8"))
        # If 'response' is a high-level object with a Body attribute
        elif hasattr(response, "body"):
            logger.info(f"{SERVE_MODE} - Reading directly from the response.body")

            body = response.body

            if str(SERVE_MODE) == "IN_PROCESS":

                # Case 1: IN_PROCESS gave [body, content_type, status]
                if isinstance(body, (list, tuple)) and len(body) == 3:
                    raw, content_type, status = body
                    # raw could be bytes or a JSON string
                    if isinstance(raw, (bytes, bytearray)):
                        obj = json.loads(raw.decode("utf-8"))
                    elif isinstance(raw, str):
                        obj = json.loads(raw)
                    else:
                        raise TypeError(f"Unexpected raw type in tuple: {type(raw)}")
                    model_output = obj["outputs"] if isinstance(obj, dict) else obj

                # Case 2: IN_PROCESS handed bytes directly
                elif isinstance(body, (bytes, bytearray)):
                    obj = json.loads(body.decode("utf-8"))
                    model_output = obj["outputs"] if isinstance(obj, dict) else obj

                # Case 3: IN_PROCESS handed a dict/list already
                elif isinstance(body, (dict, list)):
                    obj = body
                    model_output = obj["outputs"] if isinstance(obj, dict) else obj

            else:
                model_output = json.loads(body.read().decode("utf-8"))

        else:
            # If the SDK already decoded it for you (ModelBuilder sometimes does this)
            logger.info("Response directly accessible")

            model_output = response
        logger.info(f"Model input: {inputs} | Output = {model_output}")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
    finally:
        if endpoint:
            logger.info("🧹 CLEANING UP")
            try:
                if RUN_LOCAL:
                    logger.warning("Local Docker Container must be stopped manually.")

                else:
                    desc = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
                    endpoint_config_name = desc[
                        "EndpointConfigName"
                    ]  # Returns the description dict.

                    # Delete endpoint (stops compute & billing)
                    sm_client.delete_endpoint(
                        EndpointName=ENDPOINT_NAME
                    )  # Alternative way; endpoint.delete()
                    logger.info(f"Deleted endpoint '{endpoint.endpoint_name}'")

                    # Recommended to delete endpoint config and model to avoid orphaned costly resources
                    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                    sm_client.delete_model(ModelName=MODEL_NAME)
                    logger.info(f"Deleted endpoint config '{endpoint_config_name}'")
                    logger.info(f"Deleted Model '{MODEL_NAME}'")

                    logger.info("✅ Resources successfully deleted.")

            except Exception as cleanup_error:
                logger.error(f"⚠️ Cleanup failed: {cleanup_error}")
        else:
            logger.info("No endpoint was created, skipping cleanup.")


if __name__ == "__main__":
    main()
