import argparse

def add_general_group(group):
    group.add_argument("--pname", type=str, default='', help="", required=True)
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--gmode", type=str, default='clean', help="Mode of running ['clean', 'dp', 'fair', 'proposed', 'alg1', 'onebatch']")
    group.add_argument("--debug", type=int, default=1)

def add_data_group(group):
    group.add_argument('--data', type=str, default='adult', help="name of dataset")

def add_model_group(group):
    group.add_argument("--model", type=str, default='7b', help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--bs', type=int, default=512, help="batch size for training process")
    group.add_argument('--lora_r', type=int, default=16, help='number hidden embedding dim')
    group.add_argument("--epochs", type=int, default=100, help='training step')
    group.add_argument("--dout", type=float, default=0.1, help='dropout')

def parse_args():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    return parser.parse_args()
