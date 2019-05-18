#####################################################################################
# Creator     : Gaurav Roy
# Date        : 18 May 2019
# Description : The code performs APRIORI Association Rule Learning algorithm on 
#               the Market_Basket_Optimisation.csv.
#####################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Use apyori.py --> Implementation of Apriori Model, taken from Python Software Foundation

# Import Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j])
                         for j in range(0, len(dataset.columns))
                         if str(dataset.values[i, j]) != 'nan'])

# Applying Apriori ARL to the dataset
from apyori import apriori
# min_support = 3 times a day * 7 days in a week / 7501 total transactions = 0.002799 ~ 0.003
# min_confidence = Rules will be correct atleast 20% of the time
# min_lift = 3 Lifts higher than 3 are good rules
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualize the results
results = list(rules)
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE       :  ' + str(results[i][0]) + '\nSUPPORT    :  ' + str(results[i][1])
    + '\nCONFIDENCE :  ' + str(results[i][2][0][2]) + '\nLIFT       :  ' + str(results[i][2][0][3]))
