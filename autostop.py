#     Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License").
#     You may not use this file except in compliance with the License.
#     A copy of the License is located at
#
#         https://aws.amazon.com/apache-2-0/
#
#     or in the "license" file accompanying this file. This file is distributed
#     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#     express or implied. See the License for the specific language governing
#     permissions and limitations under the License.

import requests
from datetime import datetime
import getopt, sys
import urllib3
import boto3
import json
from datetime import timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Usage
usageInfo = """Usage:
This scripts checks if a notebook is idle for X seconds if it does, it'll stop the notebook:
python autostop.py --time <time_in_seconds> [--port <jupyter_port>] [--ignore-connections]
Type "python autostop.py -h" for available options.
"""
# Help info
helpInfo = """-t, --time
    Auto stop time in seconds
-p, --port
    jupyter port
-c --ignore-connections
    Stop notebook once idle, ignore connected users
-h, --help
    Help information
"""

# Read in command-line parameters
idle = True
port = "8443"
ignore_connections = False
try:
    opts, args = getopt.getopt(
        sys.argv[1:], "ht:p:c", ["help", "time=", "port=", "ignore-connections"]
    )
    if len(opts) == 0:
        raise getopt.GetoptError("No input parameters!")
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(helpInfo)
            exit(0)
        if opt in ("-t", "--time"):
            time = int(arg)
        if opt in ("-p", "--port"):
            port = str(arg)
        if opt in ("-c", "--ignore-connections"):
            ignore_connections = True
except getopt.GetoptError:
    print(usageInfo)
    exit(1)

# Missing configuration notification
missingConfiguration = False
if not time:
    print("Missing '-t' or '--time'")
    missingConfiguration = True
if missingConfiguration:
    exit(2)


def is_idle(last_activity):
    last_activity = datetime.strptime(last_activity, "%Y-%m-%dT%H:%M:%S.%fz")
    if (datetime.now() - last_activity).total_seconds() > time:
        print("Notebook is idle. Last activity time = ", last_activity)
        return True
    else:
        print("Notebook is not idle. Last activity time = ", last_activity)
        return False


def is_endpoint_idle():
    endpoint_name = get_endpoint_name()
    idle_threshold = time
    # Initialize SageMaker client
    sagemaker_client = boto3.client("sagemaker")

    # Get endpoint status
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_status = response["EndpointStatus"]

    print(f"Endpoint status: {endpoint_status}")

    # Check if the endpoint is in service
    if endpoint_status != "InService":
        return False  # Endpoint not in service, not idle

    # Get CloudWatch client
    cloudwatch_client = boto3.client("cloudwatch")

    # Get invocation metrics for the endpoint
    response = cloudwatch_client.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="Invocations",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        StartTime=(datetime.now() - timedelta(hours=1)),
        EndTime=datetime.now(),
        Period=300,
        Statistics=["Sum"],
    )

    print(response["Datapoints"])
    # Calculate total invocations in the last hour
    total_invocations = sum([datapoint["Sum"] for datapoint in response["Datapoints"]])
    print(f"Total invocations: {total_invocations}")

    # Check if total invocations are below threshold
    if int(total_invocations) == 0:
        # Check if the idle time exceeds the threshold
        last_invocation_time = max(
            [datapoint["Timestamp"] for datapoint in response["Datapoints"]]
        )
        print(f"Last invocation time: {last_invocation_time}")
        time_since_last_invocation = (
            datetime.now() - last_invocation_time
        ).total_seconds()
        print(f"Time since last invocation: {time_since_last_invocation}")
        if time_since_last_invocation >= idle_threshold:
            print("Endpoint is idle")
            return True  # Model is idle

    print("Endpoint is not idle")
    return False


def get_notebook_name():
    log_path = "/opt/ml/metadata/resource-metadata.json"
    with open(log_path, "r") as logs:
        _logs = json.load(logs)
    return _logs["ResourceName"]


def get_endpoint_name():
    return get_notebook_name() + "-model"


def stop_endpoint(endpoint_name):
    client = boto3.client("sagemaker")
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        if response["EndpointStatus"] in [
            "InService",
            "Creating",
            "Updating",
            "RollingBack",
            "SystemUpdating",
        ]:
            print("Stopping endpoint:", endpoint_name)
            client.delete_endpoint(EndpointName=endpoint_name)
            client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        else:
            print("Endpoint already stopped:", endpoint_name)
    except:
        print("Endpoint does not exist:", endpoint_name)


# This is hitting Jupyter's sessions API: https://github.com/jupyter/jupyter/wiki/Jupyter-Notebook-Server-API#Sessions-API
response = requests.get("https://localhost:" + port + "/api/sessions", verify=False)
data = response.json()
if len(data) > 0:
    for notebook in data:
        # Idleness is defined by Jupyter
        # https://github.com/jupyter/notebook/issues/4634
        if notebook["kernel"]["execution_state"] == "idle":
            if not ignore_connections:
                if notebook["kernel"]["connections"] == 0:
                    if not is_idle(notebook["kernel"]["last_activity"]):
                        idle = False
                else:
                    idle = False
                    print(
                        "Notebook idle state set as %s because no kernel has been detected."
                        % idle
                    )
            else:
                if not is_idle(notebook["kernel"]["last_activity"]):
                    idle = False
                    print(
                        "Notebook idle state set as %s since kernel connections are ignored."
                        % idle
                    )
        else:
            print("Notebook is not idle:", notebook["kernel"]["execution_state"])
            idle = False
else:
    client = boto3.client("sagemaker")
    uptime = client.describe_notebook_instance(
        NotebookInstanceName=get_notebook_name()
    )["LastModifiedTime"]
    if not is_idle(uptime.strftime("%Y-%m-%dT%H:%M:%S.%fz")):
        idle = False
        print("Notebook idle state set as %s since no sessions detected." % idle)

if idle and is_endpoint_idle():
    print("Closing idle notebook")
    client = boto3.client("sagemaker")
    notebook_name = get_notebook_name()
    # Stop the notebook instance
    print(notebook_name)
    client.stop_notebook_instance(NotebookInstanceName=notebook_name)

    # Stop the endpoint
    endpoint_name = get_endpoint_name()
    print(endpoint_name)
    stop_endpoint(endpoint_name)
else:
    print("Notebook not idle. Pass.")
