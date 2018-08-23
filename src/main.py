import argparse
import time
try:
    from src.data.data_extraction import DataExtraction
    from src.tests import *
    from src import visualize_data
    from src import tries
    from src import model_selection
except ModuleNotFoundError:
    print("Use relative import without src")
    from data.data_extraction import DataExtraction
    from tests import *
    import visualize_data
    import tries
    import model_selection

start_time = time.time()

parser = argparse.ArgumentParser(description="Predictive Analytics for Rossmann Store Sales")
parser.add_argument("action", choices=["extract", "visualize", "predict", "eval", "test", "try", "modelselection"])

args = vars(parser.parse_args())

if args["action"] == "extract":
    DataExtraction.extract()

if args["action"] == "test":
    unittest.main(argv=['first-arg-is-ignored'])

if args["action"] == "visualize":
    visualize_data.main()

if args["action"] == "try":
    tries.main()

if args["action"] == "modelselection":
    model_selection.main()


print("Finished with execution time {}s".format((time.time() - start_time)))
