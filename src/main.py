import argparse
import time
import unittest

try:
    from src.data.data_extraction import DataExtraction
    from src.tests import *
    from src import visualize_data
    from src import tries
except ModuleNotFoundError:
    print("Use relative import without src")
    from data.data_extraction import DataExtraction
    from tests import *
    import visualize_data
    import tries

start_time = time.time()

def parse_extra (parser, namespace):
  namespaces = []
  extra = namespace.extra
  while extra:
    n = parser.parse_args(extra)
    extra = n.extra
    namespaces.append(n)

  return namespaces

argparser=argparse.ArgumentParser()
subparsers = argparser.add_subparsers(help='sub-command help', dest='subparser_name')

parser_a = subparsers.add_parser('command_a', help = "command_a help")


print("Finished with execution time {}s".format((time.time() - start_time)))