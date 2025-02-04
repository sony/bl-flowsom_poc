# Copyright 2025 Sony Corporation
#
# BL-FlowSOM PoC source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

from lib.util import BatchSOM  
import numpy as np
import pandas as pd
import argparse
import random


def get_args():
    parser = argparse.ArgumentParser(description="This is a batch cluster script. ")
    parser.add_argument("-d", "--div", default=10, type=int, help="Specify som div")

    parser.add_argument(
        "-i", "--input", default="input.csv", type=str, help="Specify input csv"
    )
    parser.add_argument(
        "-o",
        "--out",
        default="clust.csv",
        type=str,
        help="Output cluster_id in each input.",
    )
    parser.add_argument(
        "-c",
        "--code",
        default="codes.csv",
        type=str,
        help="Output vectors in each node.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="seed",
    )

    return parser.parse_args()


def main():
    args = get_args()
    print("[start]read_csv")
    data = pd.read_csv(args.input, header=None)
    som_div = args.div
    grid_size = (som_div, som_div)
    print("[end]read_csv")

    if args.seed is not None:
        seed= args.seed
        random.seed(seed)
        print("set seed %d"%seed)
    
    print("[start]init_BatchSOM")
    som = BatchSOM(grid_size=grid_size, data=data, beta=0.33)  # initialize batch som
    print("[end]init_BatchSOM")
    print("[start]train_fit_BatchSOM")
    som = som.train(data, rlen=10)  # do batch som
    cl = som.fit(data)  # do clustering with trainded batch som
    print("[end]train_fit_BatchSOM")

    codes = som.get_codes()  # get each node vectors

    out_clust_file = args.out
    print("output cluster resut in %s" % out_clust_file)
    cl.to_csv(out_clust_file, header=False, index=False)
    out_codes_file = args.code
    print("output node vector resut in %s" % out_codes_file)
    codes.to_csv(out_codes_file, header=False, index=False)


if __name__ == "__main__":
    main()
