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

import os,time, sys
import multiprocessing as mp
from functools import partial

import pickle

def parallel_step_p(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability):
    # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
    result= app_defp.run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs)
    return result

def parallel_step_code(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probabilities,realization_index):

    pL_list=np.zeros((len(error_probabilities)))
    std_list=np.zeros((len(error_probabilities)))  
    pL_samples_list=np.zeros((len(error_probabilities)))
    std_samples_list=np.zeros((len(error_probabilities)))  

    # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
    for error_probability_index,error_probability in enumerate(error_probabilities):
        # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
        [[pL_list[error_probability_index],std_list[error_probability_index]],[pL_samples_list[error_probability_index],std_samples_list[error_probability_index]]] = app_defp.run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs)
    return [[pL_list,std_list] ,[pL_samples_list, std_samples_list]]

def TNDresult(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz):
    pL_list_realiz=np.zeros((num_realiz,len(error_probabilities)))
    std_list_realiz=np.zeros((num_realiz,len(error_probabilities)))      
    pL_samples_list_realiz=np.zeros((num_realiz,len(error_probabilities)))
    std_samples_list_realiz=np.zeros((num_realiz,len(error_probabilities)))      

    
    pL_list=np.zeros(len(error_probabilities))
    std_list=np.zeros(len(error_probabilities))
    pL_samples_list=np.zeros(len(error_probabilities))
    std_samples_list=np.zeros(len(error_probabilities))
    
    if code_name[:6]=='random':
        print(perm_rates)
        p=mp.Pool()
        func=partial(parallel_step_code,code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probabilities)
        result=p.map(func,range(num_realiz))
        print(result)
        p.close()
        p.join()
        for realization_index in range(num_realiz):
            for i in range(len(error_probabilities)):
                pL_list_realiz[realization_index][i]=result[realization_index][0][0][i]
                std_list_realiz[realization_index][i]=result[realization_index][0][1][i]
                pL_samples_list_realiz[realization_index][i]=result[realization_index][1][0][i]
                std_samples_list_realiz[realization_index][i]=result[realization_index][1][1][i]
        
        pL_list = np.sum(pL_list_realiz,axis=0)/num_realiz
        std_list = np.sqrt(np.sum(vsquare(std_list_realiz),axis=0))/num_realiz
        pL_samples_list = np.sum(pL_samples_list_realiz,axis=0)/num_realiz
        std_samples_list = np.sqrt(np.sum(vsquare(std_samples_list_realiz),axis=0))/num_realiz


    else:
        p=mp.Pool()
        func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name,layout)
        result=p.map(func,error_probabilities)
        print(result)
        p.close()
        p.join()
        
        for i in range(len(result)):
            pL_list[i]=result[i][0][0]   
            std_list[i]=result[i][0][1]
            pL_samples_list[i]=result[i][1][0]   
            std_samples_list[i]=result[i][1][1]

    return pL_list, std_list, pL_samples_list, std_samples_list

if __name__=='__main__':
    
    chi_val  = int(sys.argv[1])
    max_runs = int(sys.argv[2])

    def square(a):
        return a**2
    vsquare=np.vectorize(square)
    bdry_name='surface'
    code_name='random_rot_XZ_YZ'
    rotsizes= [13,17,21,25,29]
    p_min,p_max=0.05,0.50
    error_probabilities=np.linspace(p_min,p_max,50)

    bias_list=[10,100,1000,10**300]

    perm_rates=[1,0,0,0,0,0]

    for bias in bias_list:

        from itertools import cycle
        codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
        decoder = _rotatedplanarmpsdecoder_defp.RotatedPlanarMPSDecoder_defp(chi=chi_val)
        layout='rotated'
        bias_str='Z'
        num_realiz=100
        perm_rates=[1/4,1/4,1/2,0,0,0]  
                
        error_model = BiasedDepolarizingErrorModel(bias,bias_str)
        # bias=1/bias
        # error_model=BiasedYXErrorModel(bias)
        # print run parameters
        print('codes_and_size:',[code.label for code in codes_and_size])
        print('Error model:',error_model.label)
        print('number of realizations:',num_realiz)
        print('Decoder:',decoder.label)
        print('Error probabilities:',error_probabilities)
        print('Maximum runs:',max_runs)

        for L_index,code in enumerate(codes_and_size): 
            [pL_list,std_list,pL_samples_list,std_samples_list]=TNDresult(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz)

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
            results = {}
            results["logicalfailure"] = pL_list
            results["logicalfailure_std"] = std_list
            results["logicalfailure_samples"] = pL_samples_list
            results["logicalfailure_samples_std"] = std_samples_list
            output["results"] = results
            outputpath  = "data/" + code_name + "_L"+str(rotsizes[L_index]) + "_bias" + str(bias)
            outputpath += "_M" + str(max_runs) + "_chi" + str(chi_val) + ".pickle"
            fout = open(outputpath, "wb")
            pickle.dump(output, fout)
            fout.close()


