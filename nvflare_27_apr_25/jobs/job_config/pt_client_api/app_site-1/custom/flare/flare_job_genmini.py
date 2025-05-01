# flare/flare_job.py

import argparse
import os
import sys # To potentially help with imports if needed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Import user code ---
from models.classification import ClassificationNet
from run.parser import define_parser, parse_args2string # Assuming parser is in flare dir

# --- Import NVFlare components ---
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

def main():
    # --- Define local parameters ---
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    script = args.script
    key_metric = args.key_metric
    launch_process = args.launch_process
    launch_command = args.launch_command
    ports = args.ports.split(",")
    export_config = True # Set to True to generate config, False to run simulator directly (if needed)

    script_args = parse_args2string(args)

    print('\n'*3)
    print('----'*20)
    print(f"Project Root: {PROJECT_ROOT}")
    print('Args:', args)
    print('Script Args:', script_args)
    print('----'*20)

    # --- Define Initial Model ---
    # This runs when flare_job.py executes, needs access to models module
    initial_model_instance = ClassificationNet(
        feature_dim=args.feature_dim,
        classes={"TED_1": 1, "CONT_": 0}, # Example, replace with actual classes if dynamic
        lr=args.lr,
        wd=1e-6,
        loss_type=args.loss_type,
        model_architecture=args.model_architecture,
        pretrained=args.pretrained,
        use_for_classification=args.use_for_classification,
    )

    # --- Define Job ---
    job = FedAvgJob(
        name="pt_client_api", # The job folder will be named this
        n_clients=n_clients,
        num_rounds=num_rounds,
        key_metric=key_metric,
        initial_model=initial_model_instance,
    )

    # --- Assign Executor and Add Code Folders to Clients ---
    for i in range(n_clients):
        client_name = f"site-{i + 1}"

        executor = ScriptRunner(
            script=script,
            script_args=script_args,
            launch_external_process=launch_process,
            command=launch_command.replace("{PORT}", "{PORT_PLACEHOLDER}"), # Use placeholder
            framework=FrameworkType.PYTORCH,
        )
        executor.command = launch_command.replace("{PORT}", ports[i]) # Set specific port

        # Assign executor to the client application
        job.to(executor, client_name)

        # ** Add necessary code folders to this client's application **
        print(f"Adding folders to {client_name}...")
        job.to(os.path.join(PROJECT_ROOT, "dataset"), client_name)
        job.to(os.path.join(PROJECT_ROOT, "models"), client_name)
        job.to(os.path.join(PROJECT_ROOT, "util"), client_name)

        

    # --- Export Job Configuration ---
    job_output_path = "/data/michael/TED/nvflare/jobs/job_config" # Root dir for job configs
  

    if export_config:
        job.export_job(job_output_path)
        print("-" * 40)
        print(f"Job configuration exported successfully to: {job_output_path}")
        print("This configuration includes:")
        print("  - Server/Client configs (e.g., config_fed_server.json)")
        print("  - Packaged code (models/, dataset/, util/, train_fl.py)")
        print("-" * 40)
        print("Next steps:")
        print("1. (Optional) Manually edit configuration files inside:")
        print(f"   {os.path.join(job_output_path, 'app_server', 'config', 'config_fed_server.json')}")
        print("2. Run the simulation using the NVFlare CLI:")
        print(f"   nvflare simulator {job_output_path} -w /data/michael/TED/nvflare/jobs/sim_workdir -gpu 0")
        print("-" * 40)


if __name__ == "__main__":
    # It's often best to run this script from the project root directory
    # Example: python flare/flare_job.py --n_clients 2 --num_rounds 3 --script train_fl.py ... [other args]
    main()