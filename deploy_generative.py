import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# AWS and SageMaker setup
aws_region = 'us-east-1'
boto3_session = boto3.Session(profile_name='smg-group-acct', region_name=aws_region)
sm_session = sagemaker.Session(boto_session=boto3_session)

# IAM role for SageMaker execution
role = "arn:aws:iam::867344444175:role/sagemakerTaskExecutionRole"

# S3 path to model.tar.gz
model_path = "s3://smg-bucket-sandbox/DRNAICS/data/models/zephyr-7b-beta/model.tar.gz"

# HuggingFace model wrapper
huggingface_model = HuggingFaceModel(
    model_data=model_path,
    role=role,
    entry_point="inference.py",
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
    env={
        "HF_MODEL_ID": "HuggingFaceH4/zephyr-7b-beta",
        "HF_TASK": "text-generation"
    },
    sagemaker_session=sm_session
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.4xlarge",
    endpoint_name="zephyr-7b-beta-endpoint"
)

print("Endpoint deployed!")