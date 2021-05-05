import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel,BiasedYXErrorModel
from qecsim.models.planar import PlanarCode,PlanarMPSDecoder
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder

import app_defp
import _rotatedplanarmpsdecoder_defp
import importlib as imp
imp.reload(app_defp)
imp.reload(_rotatedplanarmpsdecoder_defp)

import os,time,sys
import multiprocessing as mp
from functools import partial

import pickle


def parallel_step_p(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability):
    result_onep= app_defp.run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs)
    return result_onep

def TNDresult(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz):
    p=mp.Pool()
    func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name,layout)
    result=p.map(func,error_probabilities)
    p.close()
    p.join()
    return result

def parallel_step_code(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probabilities,realization_index):
    result_one_realiz=[]
    for error_probability_index,error_probability in enumerate(error_probabilities):
        result_one_realiz.append(app_defp.run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs))
    return result_one_realiz

def TNDresult_random(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz):   
    p=mp.Pool()
    func=partial(parallel_step_code,code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probabilities)
    result=p.map(func,range(num_realiz))
    p.close()
    p.join()
    return result






def parallel_step_code2(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability,realization_index):
    result_one_realiz=app_defp.run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs)
    return result_one_realiz

def TNDresult_random2(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz):  
    result=[]
    for error_probability_index,error_probability in enumerate(error_probabilities): 
        p=mp.Pool()
        func=partial(parallel_step_code2,code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability)
        result.append(p.map(func,range(num_realiz)))
        p.close()
        p.join()
    return result


if __name__=='__main__':

    code_size = int(sys.argv[1])
    chi_val  = int(sys.argv[2])
    bias     = int(sys.argv[3])
    max_runs = int(sys.argv[4])
    
    def square(a):
        return a**2
    vsquare=np.vectorize(square)
    bdry_name='rotated'
    code_name = 'rot_XZ'

    p_min,p_max = 0.05,0.50
    error_probabilities=np.linspace(p_min,p_max,50)

    perm_rates=[1/4,1/4,1/2,0,0,0]

    from itertools import cycle
    
    code = RotatedPlanarCode(*(code_size,code_size))
    decoder = _rotatedplanarmpsdecoder_defp.RotatedPlanarMPSDecoder_defp(chi=chi_val)
    layout='rotated'
    bias_str='Z'
    num_realiz=50
            
    error_model = BiasedDepolarizingErrorModel(bias,bias_str)
    # print run parameters
    print('code:',code.label )
    print('Error model:',error_model.label)
    print('number of realizations:',num_realiz)
    print('Decoder:',decoder.label)
    print('Error probabilities:',error_probabilities)
    print('Maximum runs:',max_runs)
    
    results = TNDresult(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz)
    
    output = {}
    output['code'] = code_name
    output['error_probabilities'] = error_probabilities
    output['bias'] = bias
    output['maxruns'] = max_runs
    output['layout'] = layout
    output['chi'] = chi_val
    output['nrod'] = num_realiz
    output['bias_str'] = bias_str
    output['L'] =code_size 
    output['success_list']  = [results[k]['success_list'] for k in range(len(results))]
    output['coset_ps_list'] = [results[k]['coset_ps_list'] for k in range(len(results))] 
    output['logical_commutations_list'] = [results[k]['logical_commutations_list'] for k in range(len(results))] 
    
    outputpath  = 'data/' + code_name + '_L'+str(code_size) + '_bias' + str(bias)
    outputpath += '_M' + str(max_runs) + '_chi' + str(chi_val) + '.pickle'
    fout = open(outputpath, 'wb')
    pickle.dump(output, fout, pickle.HIGHEST_PROTOCOL)
    fout.close()


