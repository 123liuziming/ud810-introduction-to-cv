import os
import numpy as np


def read_data(inputPath, file_name):
    path = os.path.join(inputPath, file_name)
    res_lst = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            res_lst.append(list(map(lambda x: float(x), line.split())))
    return np.array(res_lst)
