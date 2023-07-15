import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import scanpy as sc
import pandas as pd
import torch
import scipy
import time
from STEM.model import *
from STEM.utils import *
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np
import anndata
from apex import amp


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map

def main():
    # parser.add_argument('--device', default='npu', type=str, help='npu or gpu')                        
    # parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')                       
    # parser.add_argument('--device_list', default='4,5,6,7', type=str, help='device id list')                  
    # parser.add_argument('--dist_backend', default='hccl', type=str,
    #                     help='distributed backend')

    print('loadings')
    scdata = anndata.read_h5ad('./data/fetal_brain/scRNA_merge_Annoed.h5ad')
    #scdata = scdata.T
    stdata = anndata.read_h5ad('./data/fetal_brain/BL_D5_lasso_cellbin.h5ad')                      #stdata = stdata.T

    sc.pp.calculate_qc_metrics(scdata,percent_top=None, log1p=False, inplace=True)
    scdata.obs['n_genes_by_counts'].median()

    sc.pp.calculate_qc_metrics(stdata,percent_top=None, log1p=False, inplace=True)
    stdata.obs['n_genes_by_counts'].median()



    # 定义条件：提取批次为 'batch_1' 的细胞
    condition = scdata.obs['batch'] == 'w15_2/5'

    # 使用布尔索引提取满足条件的细胞
    adata = scdata[condition, :]

    spoor=np.array(stdata.obs[['x','y']])
    print(spoor)

    area=float(stdata.obs[['x']].max())*float(stdata.obs[['y']].max())
    point_number=stdata.obs.shape[0]

    import math
    average_diameter=math.sqrt(area/point_number)
    # average_diameter
    print(average_diameter)

    #average_diameter=math.sqrt(np.mean(stdata.obs[['area']]))

    print('downsampling coordinate')
    # 定义掩码的尺寸
    mask_size = math.sqrt(50) * average_diameter

    new_point_i=-1
    new_mask=[]
    new_spoor = []

    # 遍历每个点的坐标
    for point_i in range(spoor.shape[0]):
        x,y=spoor[point_i] # 获取点的坐标
        #print(x,y)
        
        # 计算所属的掩码的左下角坐标
        mask_i_x = x // mask_size+2
        mask_i_y = y // mask_size+1

        if [mask_i_x,mask_i_y] not in new_spoor:
            new_spoor+=[[mask_i_x,mask_i_y]]
            new_point_i+=1
        new_mask+= [new_point_i]
        

    print('downsampling expression')
    # 应用掩码到 st据中
    new_expression=np.zeros([len(np.unique(new_mask)),stdata.X.shape[1]])
    new_spoor_reorder=np.zeros([len(np.unique(new_mask)),2])
    renew_point_i=-1
    for new_point in np.unique(new_mask):
        renew_point_i+=1
        new_expression[renew_point_i]= np.sum(stdata.X[np.array(new_mask)==new_point],axis=0)##组成新点的旧点
        new_spoor_reorder[renew_point_i][0],new_spoor_reorder[renew_point_i][1]=new_spoor[new_point]
    bin50_stdata=sc.AnnData(X=pd.DataFrame(new_expression,columns=stdata.var_names.tolist()), obsm={'spatial': new_spoor_reorder})

    spcoor=bin50_stdata.obsm['spatial']
    st_neighbor = scipy.spatial.distance.cdist(spcoor,spcoor)
    sigma = 3
    st_neighbor = np.exp(-st_neighbor**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    # 获取两个 AnnData 对象的基因名称
    genes1=adata.var_names
    genes2=bin50_stdata.var_names
    intersection = list(set(genes1) & set(genes2))
                            
    intscdata = adata[:,intersection].copy()
    intstdata = bin50_stdata[:,intersection].copy()
                            
    sc.pp.calculate_qc_metrics(intscdata,percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(intstdata,percent_top=None, log1p=False, inplace=True)
    dp = 1-intscdata.obs['n_genes_by_counts'].median()/intstdata.obs['n_genes_by_counts'].median()
    print(dp)
                            
    dp=0
                            
    sc.pp.normalize_total(intstdata)
    sc.pp.log1p(intstdata)
    sc.pp.highly_variable_genes(intstdata, n_top_genes=2000)
    # 获取高变异基因的掩码
    highly_variable_mask = intstdata.var['highly_variable']
    intstdata_top=intstdata[:,highly_variable_mask]
    sc.pp.normalize_total(intscdata)
    sc.pp.log1p(intscdata)
    intscdata_top=intscdata[:,highly_variable_mask]

    sc_adata_df=pd.DataFrame(intscdata_top.X.toarray(),index=intscdata_top.obs_names,columns=intscdata_top.var_names)
    st_adata_df=pd.DataFrame(intstdata_top.X,index=intstdata_top.obs_names,columns=intstdata_top.var_names)                           

    class setting( object ):
        pass
    seed_all(2022)
    opt= setting()
    setattr(opt, 'device', 'npu:0')
    setattr(opt, 'outf', 'log/fetal_brain_hms')
    setattr(opt, 'n_genes', sc_adata_df.shape[1])
    setattr(opt, 'no_bn', False)
    setattr(opt, 'lr', 0.002)
    setattr(opt, 'sigma', 0.5)
    setattr(opt, 'alpha', 0.8)
    setattr(opt, 'verbose', True)
    setattr(opt, 'mmdbatch', 1000)
    setattr(opt, 'dp', 0)

    testmodel = SOmodel(opt)
    testmodel.togpu()
    loss_curve = testmodel.train_wholedata(500,torch.tensor(sc_adata_df.values).float(),torch.tensor(st_adata_df.values).float(),torch.tensor(spcoor).float())

if __name__ == '__main__':
    main()
