import multiprocessing as mp
import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel
from qecsim.models.planar import PlanarCode, PlanarMPSDecoder

# set models
codes = [PlanarCode(*size) for size in [(3, 3), (5, 5)]]
error_model = PhaseFlipErrorModel()
decoder = PlanarMPSDecoder()
# set physical error probabilities
error_probability_min, error_probability_max = 0, 0.4
error_probabilities = np.linspace(error_probability_min, error_probability_max, 5)
# set max_runs for each probability
max_runs = 10

# print run parameters
print('Codes:', [code.label for code in codes])
print('Error model:', error_model.label)
print('Decoder:', decoder.label)
print('Error probabilities:', error_probabilities)
print('Maximum runs:', max_runs)


def parallel_step_p(code, error_model, decoder, max_runs,error_probability):
    return app.run(code, error_model, decoder, error_probability, max_runs=100)

pL_list =np.zeros((len(codes),len(error_probabilities)))
std_list=np.zeros((len(codes),len(error_probabilities)))

for code_index,code in enumerate(codes):
    p=mp.Pool()
    func=partial(parallel_step_p,code, error_model, decoder, max_runs)
    pL_list[code_index]=p.map(func, error_probabilities)
    p.close()
    p.join()
    