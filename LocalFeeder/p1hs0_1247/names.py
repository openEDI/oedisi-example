# -*- coding: utf-8 -*-
"""
Used to generate load csv that aligns with the GO-Solar format
"""

import numpy as np
import pandas as pd

loads = np.loadtxt('LoadsYearly.dss', delimiter= ' ', dtype='str')
load_names, load_shapes = loads[:, 1], loads[:, -1]
for i in range(len(load_names)):
    load_names[i] = load_names[i][5:]
    load_shapes[i] = load_shapes[i][7:] + '.csv'
data_dic = 'data_raw/'
Load = pd.DataFrame()
for i in range(len(load_names)):
    Load[load_names[i]] = pd.read_csv(data_dic + load_shapes[i], header=None)
Load.to_csv('loads.csv')

