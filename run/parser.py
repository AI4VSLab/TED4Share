import argparse


def define_parser():
    parser = argparse.ArgumentParser()
    # ----------------------- federated parameters ------------------
    parser.add_argument("--fl", action=argparse.BooleanOptionalAction,default = False, help="True if FL, False if local training")
    parser.add_argument("--fl_type", type=str, default="fedavg", choices=["fedavg", "fedprox", "scaffold"])
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="src/cifar10_fl.py")
    parser.add_argument("--key_metric", type=str, default="accuracy")
    parser.add_argument("--launch_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--launch_command", type=str, default="python3 -u")
    parser.add_argument("--ports", type=str, default="7777,8888")
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)

    # ----------------------- training parameters, from train.py ------------------
    parser.add_argument('--comment', type=str, default="local_model")
    parser.add_argument('--model_architecture', type=str, default="resnet50")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5) # 1e-3
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Root directory containing train.csv, val.csv, test.csv")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Checkpoint path to resume or fine-tune from. e.g. /path/to/epoch=6-step=28.ckpt")
   
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--loss_type', type=str, default="focal", choices=["focal", "ce", "simclr", "mae"], help='which loss to use')

    parser.add_argument("--use_for_classification", action=argparse.BooleanOptionalAction, default=False, required=False,
                        help="For MAE, whether to use the model for classification or not. Default is False.")
    # Inference-only argument (to test a single image at the end)
    parser.add_argument('--inference_image', type=str, default=None,
                        help="Path to a single image for testing inference. e.g. /path/to/image.png")


    return parser, parser.parse_args()

    
def parse_args2string(args, parser, ignore_args=["n_clients", "num_rounds", "script", "key_metric", "launch_process", "launch_command", "ports", "export_config"]):
    """
    Convert the arguments to a string for ScriptRunner.
    
    Args:
        args: The argument namespace from argparse
        parser: The ArgumentParser object
        ignore_args: List of argument names to ignore when converting to string
    
    Returns:
        String representation of the arguments
    """
   

    # Map dest (argument name) to action
    arg_actions = {action.dest: action for action in parser._actions}

    args_str = ""
    for arg in vars(args):
        if arg in ignore_args:
            continue
        value = getattr(args, arg)
        action = arg_actions.get(arg)
        # Check if the action is BooleanOptionalAction or value is bool
        if action and isinstance(action, argparse.BooleanOptionalAction):
            # Handle boolean flags (add only if True)
            if value:
                args_str += f" --{arg}"
        else:
            # if argument val is None, default for some args is None
            if value is not None:
                args_str += f" --{arg} {value}"
    return args_str