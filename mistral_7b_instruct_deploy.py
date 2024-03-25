# deploy_model.py
import os
from sagemaker.jumpstart.model import JumpStartModel

model_id = "huggingface-llm-mistral-7b-instruct"
model = JumpStartModel(model_id=model_id)

# Constructing endpoint name from the instance name
instance_name = os.environ.get("AWS_SAGEMAKER_NOTEBOOK_INSTANCE_NAME", "default")
endpoint_name = f"{instance_name}_mistral7binstruct_model"

# Deploying the model
predictor = model.deploy(endpoint_name=endpoint_name)
