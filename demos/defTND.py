import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel,BiasedYXErrorModel
from qecsim.models.planar import PlanarCode,PlanarMPSDecoder
# from _planarmpsdecoder_def import PlanarMPSDecoder_def
import app_def
import _planarmpsdecoder_def
import importlib as imp
imp.reload(app_def)
imp.reload(_planarmpsdecoder_def)
import os,time
import multiprocessing as mp
from functools import partial


def parallel_step_p(code,error_model,decoder,max_runs,perm_rates,code_name,error_probability):
    # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
    result= app_def.run_def(code,error_model,decoder,error_probability,perm_rates,code_name,max_runs)
    return result

def parallel_step_code(code,error_model,decoder,max_runs,perm_rates,code_name,error_probabilities,realization_index):

    pL_list=np.zeros((len(error_probabilities)))
    std_list=np.zeros((len(error_probabilities)))  

    # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
    for error_probability_index,error_probability in enumerate(error_probabilities):
        # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
        [pL_list[error_probability_index],std_list[error_probability_index]]= app_def.run_def(code,error_model,decoder,error_probability,perm_rates,code_name,max_runs)

    return [pL_list,std_list]

def square(a):
    return a**2

vsquare=np.vectorize(square)

# set models
sizes= range(6,11,2)
codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
bias_list=[10]

layout_name='planar'

code_name='spiral'
code_name='XZZX'
code_name='random'

code_name='random'
code_name='random_Hadamard'
code_name='CSS'

layout_name='planar'

if (code_name=='random' or code_name=='random_Hadamard'):
    num_realiz=40
else:
    num_realiz=1

# set physical error probabilities
error_probability_min,error_probability_max = 0.07,0.4
error_probabilities = np.linspace(error_probability_min,error_probability_max,20)

timestr = time.strftime('%Y%m%d-%H%M%S ')   #record current date and time
dirname='./data/'+timestr+layout_name+code_name
os.mkdir(dirname)     

for bias in bias_list:
    error_model = BiasedDepolarizingErrorModel(bias,'Y')
    # bias=1/bias
    # error_model=BiasedYXErrorModel(bias)
    chi_val=10
    decoder = _planarmpsdecoder_def.PlanarMPSDecoder_def(chi=chi_val)

    # set max_runs for each probability
    max_runs = 20000

    # print run parameters
    print('code_name:',code_name)
    print('codes_and_size:',[code.label for code in codes_and_size])
    print('Error model:',error_model.label)
    print('Decoder:',decoder.label)
    print('Error probabilities:',error_probabilities)
    print('Maximum runs:',max_runs)

    pL_list =np.zeros((len(codes_and_size),len(error_probabilities)))
    std_list=np.zeros((len(codes_and_size),len(error_probabilities)))

    log_pL_list =np.zeros((len(codes_and_size),len(error_probabilities)))
    log_std_list=np.zeros((len(codes_and_size),len(error_probabilities)))

    pL_list_realiz=np.zeros((len(codes_and_size),num_realiz,len(error_probabilities)))
    std_list_realiz=np.zeros((len(codes_and_size),num_realiz,len(error_probabilities)))  

    perm_rates=[0,0,0,0,0,0]

    for L_index,code in enumerate(codes_and_size):

        if code_name=='random' or code_name=='random_Hadamard':
            #permutations for random code
            if code_name=='random':
                # perm_rates=[1/4,1/4,1/4,1/4,0,0]
                perm_rates=[1/6,1/6,1/6,1/6,1/6,1/6]
            else:
                perm_rates=[1/2,1/2,0,0,0,0]

            for realization_index in range(num_realiz):
                # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
                p=mp.Pool()
                func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name)
                result=p.map(func,error_probabilities)
                #print(result)
                p.close()
                p.join()
                for i in range(len(error_probabilities)):
                    pL_list_realiz[L_index][realization_index][i]=result[i][0]
                    std_list_realiz[L_index][realization_index][i]=result[i][1]
                
            pL_list[L_index] = np.sum(pL_list_realiz[L_index],axis=0)/num_realiz
            std_list[L_index] = np.sqrt(np.sum(vsquare(std_list_realiz[L_index]),axis=0))/num_realiz

            for i in range(len(pL_list[L_index])):
                log_pL_list[L_index][i]=-np.log(pL_list[L_index][i])
                log_std_list[L_index][i]=std_list[L_index][i]/(pL_list[L_index][i]*np.log(10))

        else:
            # perm_mat,perm_vec= deform_matsvecs(code,decoder,error_model)
            p=mp.Pool()
            func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name)
            result=p.map(func,error_probabilities)
            print(result)
            p.close()
            p.join()
            
            for i in range(len(result)):
                pL_list[L_index][i]=result[i][0]   
                std_list[L_index][i]=result[i][1]
                log_pL_list[L_index][i]=-np.log(pL_list[L_index][i])
                log_std_list[L_index][i]=std_list[L_index][i]/(pL_list[L_index][i]*np.log(10))



    np.savetxt(dirname+"/p_list.csv",error_probabilities,delimiter=",")
    np.savetxt(dirname+"/pL_list.csv",pL_list,delimiter=",")
    np.savetxt(dirname+"/std_list.csv",std_list,delimiter=",")

    from itertools import cycle
    plt.figure(figsize=(20,10))
    lines = ['-',':','--','-.']
    linecycler = cycle(lines)
    plt.title('TND at bias='+str(bias)[:7]+' and chi='+str(chi_val)+' for '+layout_name+' '+code_name)
    for sizes_index,size in enumerate(sizes):
        plt.errorbar(error_probabilities,pL_list[sizes_index],std_list[sizes_index])
    plt.xlabel('p')
    plt.ylabel('$p_L$')
    plt.legend(sizes) 
    plt.savefig(dirname+'/threshold_plot_bias_'+str(bias)[:7]+'.pdf')

    from itertools import cycle
    plt.figure(figsize=(20,10))
    lines = ['-',':','--','-.']
    linecycler = cycle(lines)
    plt.title('TND failure rate scaling at bias='+str(bias)[:7]+' and chi='+str(chi_val)+' for '+layout_name+' '+code_name)
    for sizes_index,size in enumerate(sizes):
        plt.errorbar(error_probabilities,log_pL_list[sizes_index],log_std_list[sizes_index])
    plt.xlabel('p')
    plt.ylabel('$-log(p_L)$')
    plt.legend(sizes) 
    plt.savefig(dirname+'/log_scaling_plot_bias_'+str(bias)[:7]+'.pdf')

















    # if code_name=='random':

    #     for L_index,code in enumerate(codes_and_size):
    #         pH=0.5  
    #         pZY,pXY=0,0
    #         p=mp.Pool()
    #         func=partial(parallel_step_code,code,error_model,decoder,max_runs,error_probabilities)
    #         result=p.map(func,range(num_realiz))
    #         #print(result)
    #         p.close()
    #         p.join()
            
    #         for realization_index in range(num_realiz):
    #             for i in range(len(error_probabilities)):
    #                 pL_list_realiz[realization_index][L_index][i]=result[realization_index][0][i]
    #                 std_list_realiz[realization_index][L_index][i]=result[realization_index][1][i]

    #     pL_list=np.sum(pL_list_realiz,axis=0)/num_realiz
    #     std_list=np.sqrt(np.sum(vsquare(std_list_realiz),axis=0))/num_realiz

    #     for L_index,code in enumerate(codes_and_size):
    #         for i in range(len(error_probabilities)):
    #             log_pL_list[L_index][i]=-np.log(pL_list[L_index][i])
    #             log_std_list[L_index][i]=std_list[L_index][i]/(pL_list[L_index][i]*np.log(10))

    # else:
    #     for L_index,code in enumerate(codes_and_size):

    #         hadamard_mat,hadamard_vec,XYperm_mat,XYperm_vec,ZYperm_mat,ZYperm_vec= deform_matsvecs(code,decoder,error_model)
    #         p=mp.Pool()
    #         func=partial(parallel_step_p,code,hadamard_mat,error_model,decoder,max_runs)
    #         result=p.map(func,error_probabilities)
    #         print(result)
    #         p.close()
    #         p.join()
            
    #         for i in range(len(result)):
    #             pL_list[L_index][i]=result[i][0]   
    #             std_list[L_index][i]=result[i][1]
    #             log_pL_list[L_index][i]=-np.log(pL_list[L_index][i])
    #             log_std_list[L_index][i]=std_list[L_index][i]/(pL_list[L_index][i]*np.log(10))