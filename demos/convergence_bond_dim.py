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

# from _planarmpsdecoder_def import PlanarMPSDecoder_def
import app_def
import _planarmpsdecoder_def
import _rotatedplanarmpsdecoder_def
import importlib as imp
imp.reload(app_def)
imp.reload(_planarmpsdecoder_def)
imp.reload(_rotatedplanarmpsdecoder_def)
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
        print(perm_rates)
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
        func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name,layout)
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
        #     func=partial(parallel_step_p,code,error_model,decoder,max_runs,perm_rates,code_name,layout)
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

    sizes= range(13,14,2)
    rotsizes= range(13,14,2)


    #export data
    timestr=time.strftime('%Y%m%d-%H%M%S')   #record current date and time
    import os
    dirname='./data/'+'all_codes'+timestr
    os.mkdir(dirname)    #make a new directory with current date and time  

    # code_names=['spiral_XZ','random_XZ','random_XZ_YZ','random_XY']
    # code_names=['random_all','random_XZ_YZ']
    # # code_names=['spiral_XZ','random_XZ','random_all','random_XY']
    # code_names=['XY','CSS']

    bias_list=[0.5,10,100,300,1000,10**300]
    bias_list=[0.5,1,3,10,30,100,300,1000]
    
    error_probability_list=[0.189,0.194,0.222,0.278,0.335,0.39,0.428,0.456]

    bias_list=[0.5,10,100,300,1000]

    error_probability_list=[0.189,0.278,0.39,0.428,0.456]

    perm_rates=[1,0,0,0,0,0]

    bond_dimensions=[12,16,20,24,28,32]


    for bias_index,bias in enumerate(bias_list):
        error_probabilities=[error_probability_list[bias_index]]

        # code_names=['CSS','XY','XZZX','spiral_XZ','random_XZ','random_XZ_YZ']
        code_names=['random_rot_XZ_YZ']
        # from itertools import cycle
        # plt.figure(figsize=(20,10))
        # lines=['-',':','--','-.']
        # linecycler=cycle(lines)
        # plt.title('TND failure rate scaling comparison at bias='+str(bias)[:7]+' for '+layout+' '+bdry_name+'L='+str(sizes[L_index]))

        #XYZ,ZYX,XZY,YXZ,YZX,ZXY
        for code_name in code_names:
            pL_list_chi,std_list_chi,log_pL_list_chi,log_std_list_chi=np.zeros(len(bond_dimensions)),np.zeros(len(bond_dimensions)),np.zeros(len(bond_dimensions)),np.zeros(len(bond_dimensions))
            for chi_val_index,chi_val in enumerate(bond_dimensions):

                if code_name=='CSS':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    num_realiz=1
                    bias_str='Z'
                    max_runs=20000
                elif code_name=='XY':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    bias_str='Y'
                    num_realiz=1
                    max_runs=20000
                elif code_name=='rotXY':    
                    codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
                    decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)
                    layout='rotated'
                    bias_str='Y'
                    num_realiz=1
                    max_runs=10000  
                elif code_name=='rot_spiral':    
                    codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
                    decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)
                    layout='rotated'
                    bias_str='Y'
                    num_realiz=1
                    max_runs=20000  
                elif code_name=='rotXZ':    
                    codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
                    decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)
                    layout='rotated'
                    bias_str='Z'
                    num_realiz=1
                    max_runs=10000  
                elif code_name=='random_rot_XZ_YZ':    
                    codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
                    decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)
                    layout='rotated'
                    bias_str='Z'
                    num_realiz=36
                    max_runs=1000
                    perm_rates=[1/4,1/4,1/2,0,0,0]  
                elif code_name=='random_rot_XY_ZY':    
                    codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
                    decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)
                    layout='rotated'
                    bias_str='Y'
                    num_realiz=20
                    max_runs=2000  
                    perm_rates=[1/4,1/4,1/2,0,0,0]  
                elif code_name=='random_rot_XZ':    
                    codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
                    decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)
                    layout='rotated'
                    bias_str='Z'
                    num_realiz=20
                    max_runs=2000   
                    perm_rates=[1/2,1/2,0,0,0,0]  
                elif code_name=='random_rot_XZ_YZ0':    
                    codes_and_size = [RotatedPlanarCode(*(size,size)) for size in rotsizes]
                    decoder = _rotatedplanarmpsdecoder_def.RotatedPlanarMPSDecoder_def(chi=chi_val)
                    layout='rotated'
                    bias_str='Z'
                    num_realiz=30
                    max_runs=2000  
                    perm_rates=[1/4,1/3,1/3,0,0,0]                                                           
                elif code_name=='XZZX':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    num_realiz=1
                    bias_str='Z'
                    max_runs=20000
                elif code_name=='spiral_XZ':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    num_realiz=1
                    bias_str='Z'
                    max_runs=20000
                elif code_name=='random_XZ_YZ':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    num_realiz=10
                    bias_str='Z'
                    max_runs=2000
                    perm_rates=[1/2,1/3,1/2-1/3,0,0,0]
                elif code_name=='random_XZ_YZ2':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    num_realiz=10
                    bias_str='Z'
                    max_runs=2000
                    perm_rates=[1/3,1/3,1/3,0,0,0]
                elif code_name=='random_all':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    num_realiz=30
                    bias_str='Z'
                    max_runs=2000
                    perm_rates=[1/6,1/6,1/6,1/6,1/6,1/6]                    
                elif code_name=='random_XZ':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    num_realiz=30
                    bias_str='Z'
                    max_runs=2000
                    perm_rates=[1/2,1/2,0,0,0,0]
                elif code_name=='random_XY':    
                    codes_and_size = [PlanarCode(*(size,size)) for size in sizes]
                    layout='planar'
                    num_realiz=30
                    bias_str='Y'
                    max_runs=2000
                    perm_rates=[1/2,1/2,0,0,0,0]
                elif code_name=='random_ZXY':    
                    layout='planar'
                    num_realiz=30
                    bias_str='Z'
                    max_runs=2000
                    perm_rates=[1/2,0,0,0,0,1/2]
                    
                error_model = BiasedDepolarizingErrorModel(bias,bias_str)
                # bias=1/bias
                # error_model=BiasedYXErrorModel(bias)
        
                # print run parameters
                print('code_name:',code_name)
                print('codes_and_size:',[code.label for code in codes_and_size])
                print('Error model:',error_model.label)
                print('number of realizations:',num_realiz)
                print('Decoder:',decoder.label)
                print('Error probabilities:',error_probabilities)
                print('Maximum runs:',max_runs)
                
                for L_index,code in enumerate(codes_and_size):
                    [pL_list,std_list,log_pL_list,log_std_list]=TNDresult(code,decoder,error_model,max_runs,perm_rates,error_probabilities,code_name,layout,num_realiz)

                pL_list_chi[chi_val_index]=pL_list[0]
                std_list_chi[chi_val_index]=std_list[0]

            np.savetxt(dirname+'/p_list_chi'+code_name+str(bias)[:7]+'.csv',error_probabilities,delimiter=',')
            np.savetxt(dirname+'/pL_list_chi'+code_name+str(bias)[:7]+'.csv',pL_list_chi,delimiter=',')
            np.savetxt(dirname+'/std_list_chi'+code_name+str(bias)[:7]+'.csv',std_list_chi,delimiter=',')

        #     plt.errorbar(-np.log(error_probabilities),log_pL_list,log_std_list)

        # plt.xlabel('-log(p)')
        # plt.ylabel('$-log(p_L)$')
        # plt.legend(code_names) 
        # plt.savefig(dirname+'/scaling_code_comparison_bias='+str(bias)[:7]+'.pdf')


