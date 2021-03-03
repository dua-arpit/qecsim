import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel
#from qecsim.models.planar import PlanarCode, PlanarMPSDecoder
from qecsim.models.rotatedplanar import RotatedPlanarCode

from _planarmpsdecoder_def import PlanarMPSDecoder_def
from _rotatedplanarmpsdecoder_def import RotatedPlanarMPSDecoder_def
import app_def
import _rotatedplanarmpsdecoder_def
import importlib as imp
imp.reload(app_def)
imp.reload(_rotatedplanarmpsdecoder_def)
import os, time
import multiprocessing as mp
from functools import partial
from random import randint

from sklearn.utils.random import sample_without_replacement

def random_coords(dims, nsamp):
    idx = sample_without_replacement(np.prod(dims), nsamp)
    return np.vstack(np.unravel_index(idx, dims)).T


def parallel_step_p(code,hadamard_mat,error_model, decoder, max_runs,error_probability):
    result= app_def.run_def(code,hadamard_mat, error_model, decoder, error_probability, max_runs)
    return result

# set models
sizes= range(5,10,4) #choose odd sizes since we have defined hadamard_mat for odd sizes
codes_and_size = [RotatedPlanarCode(*(size,size)) for size in sizes]
bias_list=[30,1000]

code_name="optimal"
code_name="CSS"
code_name="XZZX"
code_name="random"

if (code_name=="random"):
    realizations=10
else:
    realizations=1

# set physical error probabilities
error_probability_min, error_probability_max = 0.05, 0.5
error_probabilities = np.linspace(error_probability_min, error_probability_max, 18)
# set max_runs for each probability
max_runs = 1000

biasstr_list=['X','Y','Z']

for biasstr in biasstr_list:
    timestr = time.strftime("%Y%m%d-%H%M%S ")   #record current date and time
    dirname="./data/"+timestr+code_name+'_bias='+biasstr
    os.mkdir(dirname)     

    for bias in bias_list:
        error_model = BiasedDepolarizingErrorModel(bias,'Z')
        chi_val=12
        decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)

        # print run parameters
        print('codes_and_size:', [code.label for code in codes_and_size])
        print('Error model:', error_model.label)
        print('Decoder:', decoder.label)
        print('Error probabilities:', error_probabilities)
        print('Maximum runs:', max_runs)

        pL_list_rand =np.zeros((len(codes_and_size),realizations,len(error_probabilities)))
        std_list_rand=np.zeros((len(codes_and_size),realizations,len(error_probabilities)))

        pL_list =np.zeros((len(codes_and_size),len(error_probabilities)))
        std_list=np.zeros((len(codes_and_size),len(error_probabilities)))
        
        def _rotate_q_index(index, code):
            """Convert code site index in format (x, y) to tensor network q-node index in format (r, c)"""
            site_x, site_y = index  # qubit index in (x, y)
            site_r, site_c = code.site_bounds[1] - site_y, site_x  # qubit index in (r, c)
            return code.site_bounds[0] - site_c + site_r, site_r + site_c  # q-node index in (r, c)

        for code_index,code in enumerate(codes_and_size):
            # tn_max_r, _ = _rotate_q_index((0, 0), code)
            # _, tn_max_c = _rotate_q_index((code.site_bounds[0], 0), code)

            hadamard_mat=np.zeros((sizes[code_index],sizes[code_index]))
            # n_qubits =code.n_k_d[0]
            # hadamard_mat=np.zeros(n_qubits)

            if code_name=="random":
                pH=0.5  
                for realization_index in range(realizations):
                    rand_coords= random_coords(hadamard_mat.shape,int(np.prod(hadamard_mat.shape)*pH))
                    for row, col in rand_coords:
                        hadamard_mat[row,col]=1

                    # for i in range(n_qubits):
                    #     if(np.random.rand(1,1))<pH:
                    #         hadamard_mat[i]=1

                    p=mp.Pool()
                    func=partial(parallel_step_p,code,hadamard_mat,error_model, decoder, max_runs)
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
                    for row, col in np.ndindex(hadamard_mat.shape):
                        if (row+col)%2==0:
                            hadamard_mat[row,col]=1


                if code_name=="optimal":
                    d=hadamard_mat.shape[0]
                    # for row, col in np.ndindex(hadamard_mat.shape):
                    #     if (row ==0 and col in range(0,d,4)) or (row==d-1 and col in range(2,d-1,4)) or (row%2==1 and col%2==1):
                    #         hadamard_mat[row,col]=1

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
        plt.title('TND at bias='+str(bias)+' and xi='+str(chi_val))
        for sizes_index,size in enumerate(sizes):
            plt.errorbar(error_probabilities,pL_list[sizes_index],std_list[sizes_index])
        plt.xlabel('p')
        plt.ylabel('$p_L$')
        plt.legend(sizes) 
        plt.savefig(dirname+"/threshold_plot_bias_"+str(bias)+".pdf")
