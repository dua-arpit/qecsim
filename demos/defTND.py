import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel
from qecsim.models.planar import PlanarCode, PlanarMPSDecoder
from _planarmpsdecoder_def import PlanarMPSDecoder_def
import app_def
import importlib as imp
imp.reload(app_def)
import os, time
# set models
sizes= range(5,10,2)
codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
bias=1000000
error_model = BiasedDepolarizingErrorModel(bias,'Z')
decoder = PlanarMPSDecoder_def(chi=10)
# set physical error probabilities
error_probability_min, error_probability_max = 0.3, 0.6
error_probabilities = np.linspace(error_probability_min, error_probability_max, 15)

# set max_runs for each probability
max_runs = 400

# print run parameters
print('codes_and_size:', [code.label for code in codes_and_size])
print('Error model:', error_model.label)
print('Decoder:', decoder.label)
print('Error probabilities:', error_probabilities)
print('Maximum runs:', max_runs)

import multiprocessing as mp
from functools import partial

def parallel_step_p(code,hadamard_mat,error_model, decoder, max_runs,error_probability):
    result= app_def.run_def(code,hadamard_mat, error_model, decoder, error_probability, max_runs=100)
    return result


#code_name="optimal"
code_name="random"
#code_name="XZZX"

if (code_name=="random"):
    realizations=50
else:
    realizations=1


pL_list_rand =np.zeros((len(codes_and_size),realizations,len(error_probabilities)))
std_list_rand=np.zeros((len(codes_and_size),realizations,len(error_probabilities)))

pL_list =np.zeros((len(codes_and_size),len(error_probabilities)))
std_list=np.zeros((len(codes_and_size),len(error_probabilities)))

for code_index,code in enumerate(codes_and_size):
    timestr = time.strftime("%Y%m%d-%H%M%S ")   #record current date and time
    dirname="./data/"+timestr+code_name
    os.mkdir(dirname)     
    rng = np.random.default_rng(59)

    error = error_model.generate(code, error_probability_max, rng)
    syndrome = pt.bsp(error, code.stabilizers.T)
    sample_pauli = decoder.sample_recovery(code, syndrome)
    hadamard_mat_sample=np.zeros((2*sample_pauli.code.size[0] - 1, 2*sample_pauli.code.size[1] - 1))


    if code_name=="random":
        pH=0.5  
        hadamard_mat=np.zeros(hadamard_mat_sample.shape)
        for realization_index in range(realizations):
            for row, col in np.ndindex(hadamard_mat.shape):
                if ((row%2==0 and col%2==0) or (row%2==1 and col%2==1)):
                    if(np.random.rand(1,1))<pH:
                        hadamard_mat[row,col]=1

            p=mp.Pool()
            func=partial(parallel_step_p,code,hadamard_mat, error_model, decoder, max_runs)
            result=p.map(func, error_probabilities)
            #print(result)
            p.close()
            p.join()
            for i in range(len(result)):
                pL_list_rand[code_index][realization_index][i]=result[i][0]
                std_list_rand[code_index][realization_index][i]=result[i][1]
            
        pL_list[code_index] = np.sum(pL_list_rand[code_index],axis=0)/realizations
        std_list[code_index] = np.sum(std_list_rand[code_index],axis=0)/realizations**2

    else:
        if code_name=="XZZX":
            hadamard_mat=np.zeros(hadamard_mat_sample.shape)
            for row, col in np.ndindex(hadamard_mat.shape):
                if row%2==0 and col%2==0:
                    hadamard_mat[row,col]=1

        if code_name=="optimal":
            hadamard_mat=np.zeros(hadamard_mat_sample.shape)
            d=hadamard_mat.shape[0]
            for row, col in np.ndindex(hadamard_mat.shape):
                if (row ==0 and col in range(0,d,4)) or (row==d-1 and col in range(2,d-1,4)) or (row%2==1 and col%2==1):
                    hadamard_mat[row,col]=1

        p=mp.Pool()
        func=partial(parallel_step_p,code,hadamard_mat,error_model, decoder, max_runs)
        result=p.map(func, error_probabilities)
        print(result)
        p.close()
        p.join()
        
        for i in range(len(result)):
            pL_list[code_index][i]=result[i][0]   
            std_list[code_index][i]=result[i][1]

    
from itertools import cycle
plt.figure(figsize=(20,10))
lines = ["-",":","--","-."]
linecycler = cycle(lines)
plt.title('TND at bias of'+str(bias))
for sizes_index,size in enumerate(sizes):
    plt.errorbar(error_probabilities,pL_list[sizes_index],std_list[sizes_index])
plt.xlabel('p')
plt.ylabel('$p_L$')
plt.legend(sizes) 
plt.savefig(dirname+"/threshold_plot.pdf")
