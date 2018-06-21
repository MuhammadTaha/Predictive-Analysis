import argparse
import time
# imports happen after the action is chosen, because this scripts starts faster that way

start_time = time.time()

parser = argparse.ArgumentParser(description="Predictive Analytics for Rossmann Store Sales")
parser.add_argument("action", choices=["extract", "load", "visualize", "predict", "eval", "test", "try"])

args = vars(parser.parse_args())

if args["action"] == "extract":
    from data import Data
    Data.extract()
if args["action"] == "load":
    from data import Data
    data = Data()
if args["action"] == "test":
    from tests import *
    unittest.main(argv=['first-arg-is-ignored'])
if args["action"] == "visualize":
    import visualize_data
    visualize_data.main()
if args["action"] == "try":
    import tries
    tries.main()


print("Finished with execution time {}s".format((time.time()-start_time)))