# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:20:08 2016

@author: agnus
"""

from datetime import datetime
import time
import logging
import numpy as np
import lib_sistema as ls


#import cbir_LBPSIFT_BOV as cs
#import cbir_SIFT_BOV as cs
#import cbir_SIFT as cs
#import cbir_FV as cs

# Pega a hora do sistema para identificar as diferentes execuções
DT = (datetime.now()).strftime("%Y%m%d%H%M")
#DT='201603231105'

# Define o sitema de log para registrar as ocorrências
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# create a file HANDLER
HANDLER = logging.FileHandler('./dat/sistema'+DT+'.log')
HANDLER.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
HANDLER.setFormatter(formatter)

# add the HANDLERs to the LOGGER
LOGGER.addHandler(HANDLER)

ls.grava_config()
folds, imagens, gt_filename, sift_folder, folds_folder, images_folder, subsets = ls.le_config()

t_start = time.time()
LOGGER.info('folds_construct: starting')
ls.folds_construct(subsets, folds_folder)
LOGGER.info('folds_construct: ending(' + str(time.time()-t_start)+')')

t_start = time.time()
LOGGER.info('ground_truth: starting')
gt_imagens = ls.ground_truth(folds_folder, gt_filename)
LOGGER.info('ground_truth: ending(' + str(time.time()-t_start)+')')

t_start = time.time()
LOGGER.info('gera_sift_base: starting')
ls.gera_sift_base(folds, imagens, sift_folder)
LOGGER.info('gera_sift_base: ending(' + str(time.time()-t_start)+')')

metodo = "FV_SIFT"

if metodo == "SIFT":

    t_start = time.time()
    LOGGER.info('processa_sift: starting')
    ls.processa_sift(folds, imagens, sift_folder)
    LOGGER.info('processa_sift: ending(' + str(time.time()-t_start)+')')

elif metodo == "FV_SIFT":

    # Inicialmente esta considerando apenas um fold, deve ser verifcado o caso
    # de ter mais de um fold

    n_folds = len(folds) # por enquanto n_folds será 1 pois tem apenas um fold
    for i in range(n_folds):
        train = folds[i][0]
        for j in range(n_folds):
            if j != i:
                train = train + folds[j][0]+folds[j][1]+folds[j][2]

    t_start = time.time()
    LOGGER.info('le_descritores_train: starting')
    ds, id_ds = ls.le_descritores(sift_folder, train, tipo=1)
    LOGGER.info('le_descritores_train: ending(' + str(time.time()-t_start)+')')

#%%
    N = 5  # incluir posteriormente no arquivo de configuração
    t_start = time.time()
    LOGGER.info('fv_generate_gmm: starting')
    gmm = ls.fv_generate_gmm(ds, N, DT)
    LOGGER.info('fv_generate_gmm: ending(' + str(time.time()-t_start)+')')

    #%%%

    #codifica a base em função das gmm treinadas
    t_start = time.time()
    LOGGER.info('fv_fisher_vector for train: starting')
    fv_train = np.float32([ls.fv_fisher_vector(descriptor, *gmm) for descriptor in ds])
    LOGGER.info('fv_fisher_vector for train: ending(' + str(time.time()-t_start)+')')

    #X_train = fv_train

    #%%

    #t_start = time.time()
    #LOGGER.info('FV_vetores_grava: starting')
    #ls.bov_histogramas_grava(fv_train, DT)
    #LOGGER.info('FV_vetores_grava: ending(' + str(time.time()-t_start)+')')

#%%
    # Inicialmente esta considerando apenas um fold, deve ser verifcado o caso
    # de ter mais de um fold

    n_folds = len(folds) # por enquanto n_folds será 1 pois tem apenas um fold
    for i in range(n_folds):
        test = folds[i][1]

    #ds = ls.le_descritores(sift_folder, test)
    t_start = time.time()
    LOGGER.info('le_descritores_test: starting')
    ds, id_ds = ls.le_descritores(sift_folder, test)
    LOGGER.info('le_descritores_test: ending(' + str(time.time()-t_start)+')')

#%%
    #codifica o conjunto de teste em função das gmm treinadas
    t_start = time.time()
    LOGGER.info('fv_fisher_vector for test: starting')
    fv_test = np.float32([ls.fv_fisher_vector(descriptor, *gmm) for descriptor in ds])
    LOGGER.info('fv_fisher_vector for test: ending(' + str(time.time()-t_start)+')')

    #X_test = fv_test

#%%
    import scipy.spatial.distance as ssd

    # não está considerando a questão de multiplos folds
    # e também o uso de memória auxiliar no disco

    ntrain = fv_train.shape[0]

    i = 0
    arquivo = './clist_mem_'+str(i+1)+'.txt'
    with open(arquivo, 'w') as clist_file:

        ntest = fv_test.shape[0]

        for i_test in range(ntest):

            file_test = test[i_test]
            u = fv_test[i_test]
            dist = np.zeros((ntrain))

            for i_train in range(ntrain):

                v = fv_train[i_train]
                #dist[i_train] = ssd.cityblock(u, v)
                dist[i_train] = ssd.euclidean(u, v)

            #indice = np.argsort(dist)[::-1]
            indice = np.argsort(dist)
            
            k = 1
            for idx in indice:
                clist_file.write(file_test+'|'+ str(k) +
                                 '|' + train[idx] + '|' + str(dist[idx]) +'\n')
                k = k + 1

#%%
elif metodo == "BOV":
    
#%%
    # Inicialmente esta considerando apenas um fold, deve ser verifcado o caso
    # de ter mais de um fold

    n_folds = len(folds) # por enquanto n_folds será 1 pois tem apenas um fold
    for i in range(n_folds):
        train = folds[i][0]
        for j in range(n_folds):
            if j != i:
                train = train + folds[j][0]+folds[j][1]+folds[j][2]

    t_start = time.time()
    LOGGER.info('le_descritores_train: starting')
    ds, id_ds = ls.le_descritores(sift_folder, train, 2)
    LOGGER.info('le_descritores_train: ending(' + str(time.time()-t_start)+')')

#%%
    
    k = 2000
    t_start = time.time()
    LOGGER.info('bov_codebook_gera: starting')
    centers, labels = ls.bov_codebook_gera(ds, k, 2)
    LOGGER.info('bov_codebook_gera: ending(' + str(time.time()-t_start)+')')

#%%

    t_start = time.time()
    LOGGER.info('bov_histogramas_gera: starting')    
    hists_train = ls.bov_histogramas_gera(labels, id_ds, k, train, vis=False)
    LOGGER.info('bov_histogramas_gera: ending(' + str(time.time()-t_start)+')')

#%%
    # Inicialmente esta considerando apenas um fold, deve ser verifcado o caso
    # de ter mais de um fold

    n_folds = len(folds) # por enquanto n_folds será 1 pois tem apenas um fold
    for i in range(n_folds):
        test = folds[i][1]

    #ds = ls.le_descritores(sift_folder, test)
    t_start = time.time()
    LOGGER.info('le_descritores_test: starting')
    ds, id_ds = ls.le_descritores(sift_folder, test, 2)
    LOGGER.info('le_descritores_test: ending(' + str(time.time()-t_start)+')')

    t_start = time.time()
    LOGGER.info('bov_descritores_codifica test: starting')
    labels = ls.bov_descritores_codifica(ds, centers)
    LOGGER.info('bov_descritores_codifica test: ending(' + str(time.time()-t_start)+')')        

#%%

    t_start = time.time()
    LOGGER.info('bov_histogramas_gera test: starting')    
    hists_test = ls.bov_histogramas_gera(labels, id_ds, k, test, vis=False)
    LOGGER.info('bov_histogramas_gera test: ending(' + str(time.time()-t_start)+')')

#%%
    import scipy.spatial.distance as ssd

    # não está considerando a questão de multiplos folds
    # e também o uso de memória auxiliar no disco

    ntrain = len(hists_train)

    i = 0
    arquivo = './clist_mem_'+str(i+1)+'.txt'
    with open(arquivo, 'w') as clist_file:

        ntest = len(hists_test)

        for i_test in range(ntest):

            file_test = test[i_test]
            u = hists_test[i_test]
            dist = np.zeros((ntrain))

            for i_train in range(ntrain):

                v = hists_train[i_train]
                #dist[i_train] = ssd.cityblock(u, v)
                dist[i_train] = ssd.euclidean(u, v)

            #indice = np.argsort(dist)[::-1]
            indice = np.argsort(dist)
            
            k = 1
            for idx in indice:
                clist_file.write(file_test+'|'+ str(k) +
                                 '|' + train[idx] + '|' + str(dist[idx]) +'\n')
                k = k + 1

#%%
i = 0
arquivo = './clist_mem_'+str(i+1)+'.txt'
t_start = time.time()
LOGGER.info('ground_truth: starting')
cmc = ls.compute_cmc(arquivo, gt_imagens)
LOGGER.info('ground_truth: ending(' + str(time.time()-t_start)+')')

#%%
ls.plot_cmc(cmc, len(train))
