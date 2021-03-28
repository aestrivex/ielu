#!/usr/bin/env python



from xvfbwrapper import Xvfb

vdisplay = Xvfb()
vdisplay.start()

import os
import sys
from .gselu import ElectrodePositionsModel
from . import pipeline as pipe

ct_dir = '/space/truffles/2/users/rlaplant/ct'


#1. read batch file, consisting of subject (pickle) and scope of parameters

model = ElectrodePositionsModel()
if len(sys.argv) != 3:
    raise RuntimeError('provide batch file and batch symbol as argument')

batch_file = sys.argv[1]

with open(batch_file) as fd:
    batch_data = fd.readlines()

subject, delta, rho, tau, epsilon = [x.strip() for x in batch_data]

batch_symbol = sys.argv[2]

pickles_dir = os.path.join(ct_dir, 'batch_files')
#batch_dir = os.path.join(pickles_dir, '{0}_batch'.format(batch_symbol))
output_dir = os.path.join(ct_dir, 'output_files', 
    '{0}_output'.format(batch_symbol))

pickle_file = os.path.join(pickles_dir, '{0}_batch.pkl'.format(subject))
delta = float(delta)
rho = float(rho)
tau = float(tau) / 100
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
true_grid_geom = [x for x in list(model._grid_geom.values()) if x!='user-defined']

print(true_grid_geom)

all_electrodes = list(model._all_electrodes.values())

#3. run grid algorithm with specified data and parameters

print(delta, rho, tau, epsilon)

try:
    colors, grid_geom, new_grids, color_scheme = pipe.classify_electrodes(  all_electrodes,
                                                true_grid_geom,
                                                delta = delta,
                                                epsilon = epsilon,
                                                rho = rho,
                                                crit_pct = tau,
                                                rho_strict = rho_strict,
                                                rho_loose = rho_loose )
    #if anything goes bad then set 0
except:
    new_grids = {}

#from PyQt4.QtCore import pyqtRemoveInputHook
#pyqtRemoveInputHook()
#import pdb
#pdb.set_trace()

#4. calculate correctness

electrode_scores = []

for true_grid in list(true_grids.values()):
    for elec in true_grid:

        grid_concordance = 0
        grid_size = len(true_grid)

        #find provisional grid

        provisional_grid = None
        #for new grid in new_grids:
        for new_grid in list(new_grids.values()):
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

with open(output_file, 'w') as fd:
    fd.write(str(solution_score))


if len(new_grids) > 0:

    out_pickle_file = os.path.join(output_dir, 
        '{0}_optimized_output.pkl'.format(subject))

    with open(out_pickle_file, 'w') as fd:
        model._colors = colors
        model._grid_geom = grid_geom
        model._grids = new_grids
        model._color_scheme = color_scheme

        pickle.dump(model, fd)



vdisplay.stop()
