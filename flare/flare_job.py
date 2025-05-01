# modified from: https://github.com/NVIDIA/NVFlare/blob/2.5/examples/hello-world/ml-to-fl/pt/pt_client_api_job.py
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from models.classification import ClassificationNet

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


from run.parser import define_parser, parse_args2string


def main():
    # define local parameters
    parser, args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    script = args.script
    key_metric = args.key_metric
    launch_process = args.launch_process
    launch_command = args.launch_command
    ports = args.ports.split(",")
    export_config = True #args.export_config

    script_args = parse_args2string(args, parser)

    print('\n'*3)
    print('----'*20)

    print('args:', args)
    print('script_args:', script_args)

    print('----'*20)

    

    job = FedAvgJob(
        name="pt_client_api",
        n_clients=n_clients,
        num_rounds=num_rounds,
        key_metric=key_metric,
        initial_model=ClassificationNet(
            feature_dim=args.feature_dim,
            classes = {"TED_1": 1, "CONT_": 0},
            lr=args.lr,              # can tune
            wd = 1e-6,
            loss_type=args.loss_type,     # or "ce"
            model_architecture = args.model_architecture,
            pretrained=args.pretrained,
            use_for_classification=args.use_for_classification,
        ),
    )
    
    for i in range(n_clients):
        # for args: https://nvflare.readthedocs.io/en/2.5/programming_guide/fed_job_api.html#scriptrunner
        executor = ScriptRunner(
            script=script,
            script_args =  script_args,
            launch_external_process=launch_process,
            command=launch_command.replace("{PORT}", ports[i]),
            framework=FrameworkType.PYTORCH,
        )
       
        job.to(executor, f"site-{i + 1}")
    
    if export_config:
        job.export_job("/data/michael/TED/nvflare_test/jobs/job_config")
    else:
        job.simulator_run("/data/michael/TED/nvflare/jobs/workdir", gpu="0")
    

if __name__ == "__main__":
    main()