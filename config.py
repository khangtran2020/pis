import argparse

def add_general_group(group):
    group.add_argument("--pname", type=str, default='', help="", required=True)
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--gmode", type=str, default='clean', help="Mode of running ['clean', 'dp', 'fair', 'proposed', 'alg1', 'onebatch']")
    group.add_argument("--debug", type=int, default=1)

def add_data_group(group):
    group.add_argument('--data', type=str, default='', help="name of dataset")
    group.add_argument('--label_att', type=str, default='', help="name of dataset")
    group.add_argument('--input_att', type=str, default='', help="name of dataset")
    group.add_argument('--output_att', type=str, default='', help="name of dataset")
    group.add_argument('--pperc', type=int, default=50, help="percentage of posoined data")
    group.add_argument('--rrate', type=float, default=1.0, help="reduction rate")
    group.add_argument('--prrate', type=float, default=1.0, help="reduction rate of poisoned data")
    group.add_argument('--prate', type=float, default=-1.0, help="desired poison rate")

def add_model_group(group):
    group.add_argument("--model", type=str, default='7b', help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--bs', type=int, default=512, help="batch size for training process")
    group.add_argument('--lora_r', type=int, default=16, help='number hidden embedding dim')
    group.add_argument("--epochs", type=int, default=100, help='training step')
    group.add_argument("--dout", type=float, default=0.1, help='dropout'),
    group.add_argument('--max_len', type=int, default=2048, help="model max length to use"),
    group.add_argument('--lora_a', type=int, default=32, help='lora alpha'),
    group.add_argument('--lora_dout', type=int, default=0.05, help='lora dropout'),
    group.add_argument("--temp", type=float, default=1.0, help="temperature"),
    group.add_argument('--top_k', type=int, default=50, help='top k'),
    group.add_argument("--top_p", type=float, default=0.95, help="top p")
    group.add_argument("--train", type=int, default=1, help="train or not")


def parse_args():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    return parser.parse_args()