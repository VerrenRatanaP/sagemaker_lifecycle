# deploy_model.py
import os
import json
from sagemaker.jumpstart.model import JumpStartModel

model_id = "huggingface-llm-mistral-7b-instruct"
model = JumpStartModel(model_id=model_id)


def get_notebook_name():
    log_path = "/opt/ml/metadata/resource-metadata.json"
    with open(log_path, "r") as logs:
        _logs = json.load(logs)
    return _logs["ResourceName"]


# Constructing endpoint name from the instance name
instance_name = get_notebook_name()
endpoint_name = f"{instance_name}-model"

# Deploying the model
predictor = model.deploy(endpoint_name=endpoint_name)
