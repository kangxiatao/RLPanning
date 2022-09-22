# -*- coding: utf-8 -*-

"""
Created on 06/24/2022
eval_policy_npy.
@author: AnonymousUser314156
"""

import numpy as np
import matplotlib.pyplot as plt


path = '../runs/policy_exp/TD3_2022_test2.npy'


data = np.load(path)
# print(data)
# 画图
plt.plot(data, 'r')
plt.show()

