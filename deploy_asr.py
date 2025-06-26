import boto3
import sagemaker
from sagemaker import Session
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

# AWS and SageMaker setup
aws_region = 'us-east-1'
boto3_session = boto3.Session(profile_name='smg-group-acct', region_name=aws_region)
sm_session = sagemaker.Session(boto_session=boto3_session)

# IAM role for SageMaker execution
role = "arn:aws:iam::867344444175:role/sagemakerTaskExecutionRole"

# S3 path to model.tar.gz
model_path = "s3://smg-bucket-sandbox/DRNAICS/data/models/whisper-small/model.tar.gz"

# HuggingFace model wrapper
huggingface_model = HuggingFaceModel(
    model_data=model_path,
    role=role,
    entry_point='inference.py',
    source_dir='.',
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    env={
        "HF_TASK": "automatic-speech-recognition"
    },
    sagemaker_session=sm_session
)

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=6144,
    max_concurrency=5
)

predictor = huggingface_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="whisper-small-endpoint"
)

print("Endpoint deployed!")
