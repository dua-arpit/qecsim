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
import _planarmpsdecoder_defp
import _rotatedplanarmpsdecoder_defp
import importlib as imp
imp.reload(app_defp)
imp.reload(_planarmpsdecoder_defp)
imp.reload(_rotatedplanarmpsdecoder_defp)

import os,time,sys
import multiprocessing as mp
from functools import partial

import pickle


def parallel_step_p(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability):
    # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
    result= app_defp.run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs)
    return result

def TNDresult(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz):
    pL_list_realiz=np.zeros((num_realiz,len(error_probabilities)))
    std_list_realiz=np.zeros((num_realiz,len(error_probabilities)))      
    
    pL_list=np.zeros(len(error_probabilities))
    std_list=np.zeros(len(error_probabilities))
    pL_samples_list=np.zeros(len(error_probabilities))
    std_samples_list=np.zeros(len(error_probabilities))
    coset_ps_list = []
    
    p=mp.Pool()
    func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name,layout)
    result=p.map(func,error_probabilities)
    p.close()
    p.join()
    
    #for i in range(len(result)):
    #    pL_list[i]=result[i][0][0]   
    #    std_list[i]=result[i][0][1]
    #    pL_samples_list[i]=result[i][1][0]   
    #    std_samples_list[i]=result[i][1][1]
    #    coset_ps_list.append(result[i][2])
    #return pL_list, std_list, pL_samples_list, std_samples_list, coset_ps_list
    return result

if __name__=='__main__':

    # chi_val  = int(sys.argv[1])
    # max_runs = int(sys.argv[2])
    chi_val=12
    max_runs=1000
    
    def square(a):
        return a**2
    vsquare=np.vectorize(square)
    bdry_name='surface'
    code_name = "rotXY"
    rotsizes= [13,17]
    p_min,p_max = 0.05,0.50
    error_probabilities=np.linspace(p_min,p_max,50)

    bias_list=[10]#,100,1000,10**300]

    perm_rates=[1,0,0,0,0,0]

    for bias in bias_list:

        from itertools import cycle
        
        codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
        decoder = _rotatedplanarmpsdecoder_defp.RotatedPlanarMPSDecoder_defp(chi=chi_val)
        layout='rotated'
        bias_str='Y'
        num_realiz=1
                
        error_model = BiasedDepolarizingErrorModel(bias,bias_str)
        # print run parameters
        print('codes_and_size:',[code.label for code in codes_and_size])
        print('Error model:',error_model.label)
        print('number of realizations:',num_realiz)
        print('Decoder:',decoder.label)
        print('Error probabilities:',error_probabilities)
        print('Maximum runs:',max_runs)

        for L_index,code in enumerate(codes_and_size): 
            results = TNDresult(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz)
            
            output = {}
            output["code"] = code_name
            output["error_probabilities"] = error_probabilities
            output["bias"] = bias
            output["maxruns"] = max_runs
            output["layout"] = layout
            output["chi"] = chi_val
            output["nrod"] = num_realiz
            output["bias_str"] = bias_str
            output["L"] = rotsizes[L_index]
            output["results"] = results
        
            outputpath  = "data/" + code_name + "_L"+str(rotsizes[L_index]) + "_bias" + str(bias)
            outputpath += "_M" + str(max_runs) + "_chi" + str(chi_val) + ".pickle"
            fout = open(outputpath, "wb")
            pickle.dump(output, fout, pickle.HIGHEST_PROTOCOL)
            fout.close()


