# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:41:13 2023

@author: 12036
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

x_train = ["Hit me up", "Let's get on a game", "Unless you are busy"]
from sklearn.feature_extraction.text import CountVectorizer

Z = CountVectorizer()
Z.fit(x_train)
Z.get_feature_names_out()

print(Z)