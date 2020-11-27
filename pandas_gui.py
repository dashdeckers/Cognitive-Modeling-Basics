import argparse
from pathlib import Path

import pandas as pd
from pandasgui import show

parser = argparse.ArgumentParser(
    description='Start PandasGUI',
    epilog='Provide the directory name containing a csv, or a csv file directly'
)
parser.add_argument(
    '-f', '--file', type=str,
    help='directory name containing- or the path to a .csv file',
    default='HW2'
)
args = parser.parse_args()


if __name__ == '__main__':
    if args.file.endswith('.csv'):
        show(pd.read_csv(args.file))
    else:
        for filepath in (Path() / args.file).rglob('*.csv'):
            show(pd.read_csv(filepath))
