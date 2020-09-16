import argparse
import os


def run_bash(cmd: str):
    os.system(cmd)


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
                    help="choose one of the models: logistic_regression, GRU, external_GRU, transformers")
args = parser.parse_args()
if args.model == "logistic_regression":
    run_bash("python3 train_logistic_classifer.py")
elif args.model == "GRU":
    run_bash("python3 train_gru.py")
elif args.model == "external_GRU":
    run_bash("python3 readGlov.py")
    run_bash("python3 train_externalgru.py")
elif args.model == "transformers":
    run_bash("python3 train_transformer.py")
else:
    print("wrong choice, please check your command")
