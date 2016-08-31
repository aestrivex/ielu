#!/usr/bin/env python

from __future__ import division

import os
import sys
from gselu import ElectrodePositionsModel
import pipeline as pipe

pickles_dir = '/home/blackreach/subjects/pickle_files'
output_dir = '/home/blackreach/subjects/output_files'

#1. read batch file, consisting of subject (pickle) and scope of parameters

model = ElectrodePositionsModel()
if len(sys.argv) != 2:
    raise RuntimeError('provide batch file as argument')

batch_file = sys.argv[1]

with open(batch_file) as fd:
    batch_data = fd.readlines()

subject, delta, rho, tau, epsilon = map(lambda x:x.strip(), batch_data)


pickle_file = os.path.join(pickles_dir, '{0}_batch.pkl'.format(subject))
delta = float(delta)
rho = float(rho)
tau = float(tau)
epsilon = float(epsilon)

if rho < 15:
    rho_strict = 5
    rho_loose = rho + 15
elif rho > 75:
    rho_strict = rho - 15
    rho_loose = 85
else:
    rho_strict = rho - 15
    rho_loose = rho + 15
    

#2. load pickle data

from pickle import load

with open(pickle_file) as fd:
    model = load(fd)

true_grids = model._grids
true_grid_geom = filter(lambda x:x!='user-defined', model._grid_geom.values())

print true_grid_geom

all_electrodes = model._all_electrodes.values()

#3. run grid algorithm with specified data and parameters

_, _, new_grids, _ = pipe.classify_electrodes(  all_electrodes,
                                                true_grid_geom,
                                                delta = delta,
                                                epsilon = epsilon,
                                                rho = rho,
                                                crit_pct = tau,
                                                rho_strict = rho_strict,
                                                rho_loose = rho_loose )

#4. calculate correctness

electrode_scores = []

for true_grid in true_grids:
    for elec in true_grid:

        grid_concordance = 0
        grid_size = len(true_grid)

        #find provisional grid

        provisional_grid = None
        for new_grid in new_grids:
            if elec in new_grid:
                provisional_grid = new_grid
                break

        #if we failed to find the electrode in the true solution this
        #electrode is assigned a score of 0
        if provisional_grid is None:
            score = 0
            electrode_scores.append(score)
            break 

        # calculate completeness
        for reference_elec in true_grid:
            if reference_elec in provisional_grid:
                grid_concordance += 1
        
        score = grid_concordance / grid_size
        electrode_scores.append(score)

nr_electrodes = len(electrode_scores)
solution_score = sum(electrode_scores) / nr_electrodes

#5. save results to disk

output_file = os.path.join(output_dir, '{0}_{1}_{2}_{3}_{4}.output'.format(
                    subject,
                    delta,
                    rho,
                    tau,
                    epsilon))

with open('output_file', 'w') as fd:
    fd.write(str(solution_score))
