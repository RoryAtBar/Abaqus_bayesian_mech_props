#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:31:56 2024

@author: w10944rb
"""

import numpy as np
import pandas as pd
from sys import argv
from scipy.stats import qmc


c0_min = float(argv[1])
c0_max = float(argv[2])
c1_min = float(argv[3])
c1_max = float(argv[4])
c2_min = float(argv[5])
c2_max = float(argv[6])
c3_min = float(argv[7])
c3_max = float(argv[8])
c4_min = float(argv[9])
c4_max = float(argv[10])
c5_min = float(argv[11])
c5_max = float(argv[12])
runs = int(argv[13])
friction = float(argv[14])
conductance = [float(argv[15]), argv[16]]
power = float(argv[17])


hypercube_obj = qmc.LatinHypercube(d=6)
samples = hypercube_obj.random(runs)
lower_bounds = [c0_min,c1_min,c2_min,c3_min,c4_min,c5_min]
upper_bounds = [c0_max,c1_max,c2_max,c3_max,c4_max,c5_max]

scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

print(scaled_samples[:,0])

cond_input = []

for i in range(runs):
    cond_input.append(conductance[1])


results = {'C0':scaled_samples[:,0], 
           'C1':scaled_samples[:,1], 
           'C2':scaled_samples[:,2], 
           'C3':scaled_samples[:,3], 
           'C4':scaled_samples[:,4],
           'C5':scaled_samples[:,5],
           'Friction':np.ones(runs)*friction,
           'Conductance': np.ones(runs)*conductance[0],
           'Conductance input file': cond_input,
           'Power': np.ones(runs)* power,
           'Force Results1':np.zeros(runs), 
           'Force Results2':np.zeros(runs),
           'PEEQ Results':np.zeros(runs), 
           'Barrelling Profile':np.zeros(runs),
           'Temperature profile':np.zeros(runs)}

results_df = pd.DataFrame(results)
results_df['Force Results1'] = results_df['Force Results1'].astype(object)
results_df['Force Results2'] = results_df['Force Results2'].astype(object)
results_df['Barrelling Profile'] = results_df['Barrelling Profile'].astype(object)
results_df['PEEQ Results'] = results_df['PEEQ Results'].astype(object)
results_df['Temperature profile'] = results_df['Temperature profile'].astype(object)

results_df.to_pickle('Quad_plasticity.pkl')
