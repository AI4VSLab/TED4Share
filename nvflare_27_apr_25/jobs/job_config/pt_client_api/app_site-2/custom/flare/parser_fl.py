import argparse


def define_parser():
    parser = argparse.ArgumentParser()
    # ----------------------- federated parameters ------------------
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
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
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


    return parser.parse_args()

def parse_args2string(args):
    """
    Convert the arguments to a string for ScriptRunner.
    """
    args_str = ""
    for arg in vars(args):
        args_str += f" --{arg} {getattr(args, arg) }"
    return args_str