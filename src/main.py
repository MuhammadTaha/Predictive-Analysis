import argparse
import time
# imports happen after the action is chosen, because this scripts starts faster that way
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()
start_time = time.time()

parser = argparse.ArgumentParser(description="Predictive Analytics for Rossmann Store Sales")
parser.add_argument("action", choices=["extract", "load", "visualize", "predict", "eval", "test"])

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
    import visualize
    visualize.main()

print("Finished with execution time {}s".format((time.time()-start_time)))