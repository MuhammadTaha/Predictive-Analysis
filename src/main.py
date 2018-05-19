import argparse
import pdb


parser = argparse.ArgumentParser(description="Predictive Analytics for Rossmann Store Sales")
parser.add_argument("action", choices=["extract", "visualize", "predict", "eval"])

args = vars(parser.parse_args())

if args["action"] == "extract":
    print("Extract .zip")
