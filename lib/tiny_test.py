# Copyright 2025 Sony Corporation
#
# BL-FlowSOM PoC source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

from util import BatchSOM  # type: ignore
import numpy as np
import pandas as pd


def main():
    np.random.seed(seed=2025)  # set random seed
    data = np.random.rand(110, 3)  # 110 events 3dim data
    grid_size = (4, 4)  # set som size 4x4
    som = BatchSOM(grid_size=grid_size, data=data, beta=0.33)  # initialize batch som
    som = som.train(data, rlen=10)  # do batch som
    cl = som.fit(data)  # do clustering with trainded batch som
    codes = som.get_codes()  # get each node vectors
    print("output cluster resut in clus.csv")
    cl.to_csv("clus.csv", header=False, index=False)
    print("output node vector resut in codes.csv")
    codes.to_csv("codes.csv", header=False, index=False)


if __name__ == "__main__":
    main()
