import argparse
import pdb

from data import Data


parser = argparse.ArgumentParser(description="Predictive Analytics for Rossmann Store Sales")
parser.add_argument("action", choices=["extract", "load", "visualize", "predict", "eval", "test"])

args = vars(parser.parse_args())

if args["action"] == "extract":
    Data.extract()
if args["action"] == "load":
    data = Data()
if args["action"] == "test":
    from tests import *
    unittest.main(argv=['first-arg-is-ignored'])