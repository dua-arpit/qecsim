import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel,BiasedYXErrorModel
from qecsim.models.planar import PlanarCode,PlanarMPSDecoder
import os,time
import multiprocessing as mp
from functools import partial



def parallel_step_p(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability):
    # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
    result= app_def.run_def(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs)
    return result

def parallel_step_code(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probabilities,realization_index):

    pL_list=np.zeros((len(error_probabilities)))
    std_list=np.zeros((len(error_probabilities)))  

    # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
    for error_probability_index,error_probability in enumerate(error_probabilities):
        # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
        [pL_list[error_probability_index],std_list[error_probability_index]]= app_def.run_def(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs)

    return [pL_list,std_list]

def TNDresult(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz):
    pL_list_realiz=np.zeros((num_realiz,len(error_probabilities)))
    std_list_realiz=np.zeros((num_realiz,len(error_probabilities)))      
    
    pL_list=np.zeros(len(error_probabilities))
    std_list=np.zeros(len(error_probabilities))
    log_pL_list=np.zeros(len(error_probabilities))
    log_std_list=np.zeros(len(error_probabilities))
    
    if code_name[:6]=='random':
        p=mp.Pool()
        func=partial(parallel_step_code,code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probabilities)
        result=p.map(func,range(num_realiz))
        #print(result)
        p.close()
        p.join()
        for realization_index in range(num_realiz):
            for i in range(len(error_probabilities)):
                pL_list_realiz[realization_index][i]=result[realization_index][0][i]
                std_list_realiz[realization_index][i]=result[realization_index][1][i]
        
        pL_list = np.sum(pL_list_realiz,axis=0)/num_realiz
        std_list = np.sqrt(np.sum(vsquare(std_list_realiz),axis=0))/num_realiz

        for i in range(len(pL_list)):
            log_pL_list[i]=-np.log(pL_list[i])
            log_std_list[i]=std_list[i]/(pL_list[i]*np.log(10))

    else:
        p=mp.Pool()
        func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name)
        result=p.map(func,error_probabilities)
        print(result)
        p.close()
        p.join()
        
        for i in range(len(result)):
            pL_list[i]=result[i][0]   
            std_list[i]=result[i][1]
            log_pL_list[i]=-np.log(pL_list[i])
            log_std_list[i]=std_list[i]/(pL_list[i]*np.log(10))

        # for realization_index in range(num_realiz):
        #     p=mp.Pool()
        #     func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name)
        #     result=p.map(func,error_probabilities)
        #     print(result)
        #     p.close()
        #     p.join()

        #     for i in range(len(error_probabilities)):
        #         pL_list_realiz[realization_index][i]=result[i][0]
        #         std_list_realiz[realization_index][i]=result[i][1]

        # pL_list = np.sum(pL_list_realiz,axis=0)/num_realiz
        # std_list = np.sqrt(np.sum(vsquare(std_list_realiz),axis=0))/num_realiz  

        # for i in range(len(pL_list)):
        #     log_pL_list[i]=-np.log(pL_list[i])
        #     log_std_list[i]=std_list[i]/(pL_list[i]*np.log(10))

    return [pL_list,std_list,log_pL_list,log_std_list]


if __name__=='__main__':

    def square(a):
        return a**2
    vsquare=np.vectorize(square)
    layout='planar'
    bdry_name='surface'

    sizes= range(6,7,2)
    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
    p_min,p_max=0.01,0.50
    error_probabilities=np.linspace(p_min,p_max,8)

    #export data
    # code_names=['spiral_XZ','random_XZ','random_XZ_YZ','random_XY']
    # code_names=['random_all','random_XZ_YZ']
    # # code_names=['spiral_XZ','random_XZ','random_all','random_XY']
    # code_names=['XY','CSS']

    bias_list=[100]

    code_names=['XY','random_XZ_YZ4','random_XZ_YZ3','random_XZ_YZ2','random_XZ_YZ1','random_XZ_YZ0']

    perm_rates=[1,0,0,0,0,0]
    chival=13

    for L_index,code in enumerate(codes_and_size):
        for bias in bias_list:
            from itertools import cycle
            plt.figure(figsize=(20,10))
            lines=['-',':','--','-.']
            linecycler=cycle(lines)
            plt.title('TND failure rate scaling comparison at bias='+str(bias)[:7]+' for '+layout+' '+bdry_name+', L='+str(sizes[L_index])+', chi='+str(chival))

            #XYZ,ZYX,XZY,YXZ,YZX,ZXY
            for code_name in code_names:                                   
                error_probabilities=np.loadtxt("p_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                pL_list=np.loadtxt("pL_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                std_list=np.loadtxt("std_list"+code_name+str(bias)[:7]+".csv",delimiter=",")
                log_pL_list,log_std_list=pL_list,std_list
                for i in range(len(pL_list)):
                    log_pL_list[i]=-np.log(pL_list[i])
                    log_std_list[i]=std_list[i]/(pL_list[i]*np.log(10))
                plt.errorbar(-np.log(error_probabilities),log_pL_list,log_std_list)

            plt.xlabel('-log(p)')
            plt.ylabel('$-log(p_L)$')
            plt.legend(code_names) 
            plt.savefig('scaling_code_comparison_bias='+str(bias)[:7]+'.pdf')


