# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:36:05 2016

@author: agnus
"""
#%%
def monta_lista_imagens(path = '.', ext='.png'):
    import os

    imagens = {}
    for dirname, dirnames, filenames in os.walk(path):
        # print path to all filenames with extension py.
        for filename in filenames:
            fname_path = os.path.join(dirname, filename)
            fext = os.path.splitext(fname_path)[1]
            if fext == ext:
                #file_dat = [filename, dirname]
                #imagens.append(file_dat)
                imagens[filename]=dirname
            else:
                continue

    return imagens

#%%
def grava_db_imagens(arquivo, imagens):
    #arquivo = './tatt_c.db'
    with open(arquivo, 'wb') as db_image_file:
        for  nome_img, caminho in imagens.items():
            db_image_file.write(nome_img+ '\t' + caminho + '\n')
        db_image_file.close()

#%%
def grava_config(arquivo = './example_mem.cfg'):

    import ConfigParser

    config = ConfigParser.RawConfigParser()

    # When adding sections or items, add them in the reverse order of
    # how you want them to be displayed in the actual file.
    # In addition, please note that using RawConfigParser's and the raw
    # mode of ConfigParser's respective set functions, you can assign
    # non-string values to keys internally, but will receive an error
    # when attempting to write to a file or when you get it in non-raw
    # mode. SafeConfigParser does not allow such assignments to take place.

    config.add_section('Geral')
    config.set('Geral', 'Image Database', 'Tatt-C')
    config.set('Geral', 'Database Image Folder', '/media/sf_Projeto/dataset/tatt_dca/')
    config.set('Geral', 'Indexa image database', 'True')
    config.set('Geral', 'Database filename', './tatt_c.db')
    config.set('Geral', 'Image filename extension','.jpg')

    config.set('Geral', 'Training File', 'train1')
    config.set('Geral', 'Testing File', 'test1')

    config.add_section('Folds')
    config.set('Folds', 'Folds Folder', '/media/sf_Projeto/dataset/tatt_dca/folds/')
    config.set('Folds', 'Quantidade subsets', '3')
    config.set('Folds', 'Subset_1', 'gallery{1}.txt')
    config.set('Folds', 'Subset_2', 'probes{1}.txt')
    config.set('Folds', 'Subset_3', 'bg{1}.txt')
    config.set('Folds', 'Ground_truth', 'ground_truth.txt')
    
    config.add_section('SIFT')
    config.set('SIFT','SIFT Folder',  '/media/sf_Projeto/dataset/tatt_dca/SIFT/')

    # Writing our configuration file to 'example.cfg'
    with open(arquivo, 'wb') as configfile:
        config.write(configfile)


#%%
def folds_construct(subsets, folds_folder):
    
    n_folds =len(subsets[0])
    n_subsets = len(subsets)
    folds = []
    for i in range(n_folds):
        sub = []
        for j in range(n_subsets):
            arquivo = subsets[j][i]
            aux = []
            with open(folds_folder+arquivo, 'r') as imagefiles:
                for nomef in imagefiles:
                    if nomef[-1] == '\n' : nomef = nomef[:-1]
                    aux.append(nomef)
            imagefiles.close()
            sub.append(aux)
        folds.append(sub)
    
    return folds

#%%
def le_config():
    import ConfigParser

    config = ConfigParser.RawConfigParser()
    config.read('./example_mem.cfg')

    # getfloat() raises an exception if the value is not a float
    # getint() and getboolean() also do this for their respective types
    base = config.get('Geral', 'image database')
    indexa = config.getboolean('Geral', 'indexa image database')
    print base
    if indexa:
        print "indexa base"
        arquivo = config.get('Geral','database filename')
        caminho = config.get('Geral', 'database image folder')
        extensao = config.get('Geral', 'image filename extension')

        print arquivo, caminho, extensao
        imagens = monta_lista_imagens(caminho, extensao)

        grava_db_imagens(arquivo, imagens)

    folds_folder = config.get('Folds','folds folder')
    n_subsets = config.getint('Folds', 'quantidade subsets')

    subsets=[]

    for i in range(n_subsets):
        sub = config.get('Folds', 'subset_'+str(i+1))
        ps = sub.find("{")
        pe = sub.find("}")
        ped = sub[ps+1:pe]
        indices = ped.split(',')
        aux = []
        for ind in indices:
            aux.append(sub[:ps]+ind+'.txt') # incluir extensão variável
        subsets.append(aux)

    #print subsets

    #n_folds = config.getint('Folds', 'quantidade folds')
    n_folds =len(subsets[0])
    folds = []
    for i in range(n_folds):
        sub = []
        for j in range(n_subsets):
            arquivo = subsets[j][i]
            aux = []
            with open(folds_folder+arquivo, 'r') as imagefiles:
                for nomef in imagefiles:
                    if nomef[-1] == '\n' : nomef = nomef[:-1]
                    aux.append(nomef)
            imagefiles.close()
            sub.append(aux)
        folds.append(sub)

    #print folds[0]

    gt_filename = config.get('Folds', 'ground_truth')

    sift_folder = config.get('SIFT', 'sift folder')
    
    print sift_folder, folds_folder, caminho
    
    return (folds, imagens, gt_filename, sift_folder, folds_folder, caminho, subsets)
           
#%%
def sift(nomes_imagens, imagens, sift_folder):

    import cv2
    import os
    from math import sqrt

    #ds = []
    #kp = []
    t = len(nomes_imagens)
    i=1


    for filename in nomes_imagens:

        fname = os.path.join(sift_folder, filename[:-3]+'sift_ds')

        if os.path.isfile(fname) == False :
            print filename
            #file_img = os.path.join(diretorio, filename)
            diretorio = imagens[filename]
            img = cv2.imread(os.path.join(diretorio, filename)) #file_img)
            # Redimensiona imagem para aplicação do Fisher Vectors
            #img = cv2.resize(img, (256,256))
            aux = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(aux)
            k = sqrt((240.0*480.0*0.5)/(gray.shape[0]*gray.shape[1]))
            res = cv2.resize(gray,None,fx=k, fy=k, interpolation = cv2.INTER_CUBIC)
            sift = cv2.xfeatures2d.SIFT_create()
            (kps, descs) = sift.detectAndCompute(res, None)

            #ds.append(descs)
            #kp.append(kps)

            arquivo = os.path.join(sift_folder, filename[:-3]+'sift_ds')
            with open(arquivo, 'wb') as sift_file:
                for desc in descs:
                    sift_file.write(','.join(str(x) for x in desc)+'\n')
                sift_file.close()

            arquivo = os.path.join(sift_folder, filename[:-3]+'sift_kp')
            with open(arquivo, 'wb') as sift_file:
                for point in kps:
                    temp = [point.pt[0], point.pt[1], point.size, point.angle,
                        point.response, point.octave, point.class_id]
                    sift_file.write(','.join(str(x) for x in temp)+'\n')
                sift_file.close()
    
        print (i*100)/t,
        i=i+1

    #return ds

#%%
def sift_match(ds1, kp1, ds2, kp2):

    import cv2

    MIN_MATCH_COUNT = 10
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ds1,ds2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    qm = len(good)

    (nr1,c) = ds1.shape
    (nr2,c) = ds2.shape
    
    #    if qm>MIN_MATCH_COUNT:
    #        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #    
    #        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #        if mask != None:
    #            matchesMask = mask.ravel().tolist()
    #            rt = np.sum(np.asarray(matchesMask))
    #        else:
    #            rt = 0
    #    else:
    #        #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    #        #matchesMask = None
    #        rt = 0

    nr = nr1
    if nr2>nr:
        nr = nr2

    rt = (100.0*qm/nr)
    #    if qm > 0:
    #        rt = 1.0/qm
    #    else:
    #        rt = 10^8

    return rt

#%%
def gera_sift_base(folds, imagens, sift_folder):
    
    # Inicialmente gera se necessario o SIFT para as imagens de treinamento e teste
    # pode ser otimizado, gerando para toda a base, caso se utilize toda a base
    # o que pode ter um custo alto pois na base existem imagens para outros casos
    # de uso.
    n_folds = len(folds)
    #Poder ser implementado diferente pois as linhas abaixo apenas agregram os nomes
    #das imagens para que sejam gerados os sifts para cada um dos folds
    for i in range(n_folds):
        test = folds[i][1]
        train = folds[i][0]
        bg = folds[i][2]
        for j in range(n_folds):
            if j!=i :
                train = train + folds[j][0]+folds[j][1]+folds[j][2]
    
        print 'Gerando sift do conjunto de treinamento'
        #train_kp, train_ds = sift(train, imagens, sift_folder)
        sift(train, imagens, sift_folder)
        print 'Gerando sift do conjunto de teste'
        #test_kp, test_ds = sift(test, imagens)
        sift(test, imagens, sift_folder)
        print 'Gerando sift do conjunto de bg'
        #bg_kp, bg_ds = sift(bg, imagens)
        sift(bg, imagens, sift_folder)
        
#%%
def processa_sift(folds, imagens, sift_folder):
    
    import numpy as np
    import os
    import cv2
    
    n_folds = len(folds)
    #Alterei para que inclua nas imagens da galeria i no conj. train, de forma a que as
    # imagens correspondentes ao probe existam na galeria (train)
    for i in range(n_folds):
        test = folds[i][1]
        bg = folds[i][2]
        train = folds[i][0]#+bg
        for j in range(n_folds):
            if j!=i :
                train = train + folds[j][0]+folds[j][1]+folds[j][2]
    
        n_test = len(test)
        n_train = len(train)
    
        dist = np.zeros((n_train), dtype=np.float)
        nn = n_test * n_train
    
        print 'Gerando o match entre o treinamento e o conjunto de teste'
    
        mem = True
        if mem==True :
            ds=[]
            ks=[]
            
        arquivo = './clist_mem_'+str(i+1)+'.txt'
        with open(arquivo, 'w') as clist_file:
    
            l = 0
            
            for file_test in test:
        
                fname = os.path.join(sift_folder, file_test[:-3]+'sift_ds')
                ds1 = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.uint8) #,skiprows=1)
                fname = os.path.join(sift_folder, file_test[:-3]+'sift_kp')
                kps = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.float) #,skiprows=1)
                
                kp1=[]
                kp2=[]
                for kp in kps:
                    kpoint = cv2.KeyPoint(float(kp[0]), float(kp[1]),
                                      float(kp[2]), float(kp[3]),
                                      float(kp[4]), int(kp[5]), int(kp[6]))
                    kp1.append(kpoint)            
                
                diretorio = imagens[file_test]
                img1 = cv2.imread(os.path.join(diretorio, file_test),0) 
                #print os.path.join(diretorio, file_test)
                j = 0
        
                for file_train in train:
                    diretorio = imagens[file_train]
                    img2 = cv2.imread(os.path.join(diretorio, file_train),0)
                    #print os.path.join(diretorio, file_train)
                    if (mem == True and len(ds)<len(train)):
                        fname = os.path.join(sift_folder, file_train[:-3]+'sift_ds')
                        ds.append ( np.asarray((np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.uint8)) ) #,skiprows=1)
                        ds2 = ds[j]
                        
                        fname = os.path.join(sift_folder, file_train[:-3]+'sift_kp')
                        kps = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.float) #,skiprows=1)
                
                        aux =[]
                        for kp in kps:
                            kpoint = cv2.KeyPoint(float(kp[0]), float(kp[1]),
                                      float(kp[2]), float(kp[3]),
                                      float(kp[4]), int(kp[5]), int(kp[6]))
                            aux.append(kpoint)
                        ks.append(aux)
                        kp2 = ks[j]
    
                    elif (mem == True and len(ds)==len(train)):
                        ds2 = ds[j]
                        kp2 = ks[j]
                    elif mem == False:
                        fname = os.path.join(sift_folder, file_train[:-3]+'sift_ds')
                        ds2 = ( (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.uint8) )
                        
                        fname = os.path.join(sift_folder, file_train[:-3]+'sift_kp')
                        kps = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.float) #,skiprows=1)
                        
                        kp2 = []
                        for kp in kps:
                            kpoint = cv2.KeyPoint(float(kp[0]), float(kp[1]),
                                      float(kp[2]), float(kp[3]),
                                      float(kp[4]), int(kp[5]), int(kp[6]))
                            kp2.append(kpoint)  
                    #print ds1
                    #print ds2
                    
                    rt = sift_match(ds1, np.asarray(kp1), ds2, np.asarray(kp2))
                    dist[j] = rt
                    j = j + 1
                    print i,(((l*n_train)+j)*100)/nn,

                indice = np.argsort(dist)[::-1]
                k = 1
                for id in indice:
                    clist_file.write(file_test+'|'+ str(k) + '|' + train[id] + '|' + str(dist[id]) +'\n')
                    k = k + 1
                    
                l = l + 1
                
            clist_file.close()
                
            break    

#%%

def ground_truth(folds_folder, gt_filename):
    """Reads a ground truth table from text file.

    Keyword arguments:
    folds_folder -- the path for the ground truth file
    gt_filename -- the file name of the ground truth file with extension
    
    Returns:
    gt_images -- ground truth table stored in a dictionary
    """    
    
    #folds_folder = '/media/sf_Projeto/dataset/tatt-c_update_v1.4/5-fold/tattoo_identification/'
    #gt_filename = 'ground_truth.txt'
    gt_imagens = {}
    with open(folds_folder+gt_filename, 'r') as gt_arq:
        for nomef in gt_arq:
            imgs = nomef.split('|')
            if imgs[1][-1] == '\n' : imgs[1] = imgs[1][:-1]
            #print imgs[0], imgs[1]
            gt_imagens[imgs[0]] = imgs[1]
        gt_arq.close()
        
    return gt_imagens

#%%
def compute_cmc(arquivo, gt_imagens):
    """Reads a classification list from text file and sumarize rank results for
        every image reference based in the ground truth dictionary.

    Keyword arguments:
    arquivo -- the filename of classification list file
    gt_images -- ground truth table stored in a dictionary
    
    Returns:
    cmc -- acummulated accuracy for each rank stored in a numpy array
    """    
    import numpy as np
    i = 0
    acc = np.zeros(400)
    #arquivo = './clist_mem_'+str(i+1)+'.txt'
    with open(arquivo, 'r') as clist_file:  
        for nomef in clist_file:
            imgs = nomef.split('|')
            if imgs[3][-1] == '\n' : imgs[3] = imgs[3][:-1]
            if gt_imagens[imgs[0]] == imgs[2] :
                r = int(imgs[1])
                acc[r] = acc[r]+1
    clist_file.close()
    
    #print cmc
    ft = sum(acc)
    #print cmc/ft
    cmc = np.zeros(400)
    for i in range(1,400):
        cmc[i] = cmc[i-1]+acc[i]/ft
    #print cmc1 
    
    return cmc

#%%
def plot_cmc(cmc):
    
    import matplotlib.pyplot as plt
    import pylab as P
    import numpy as np
    
    fig = P.figure()
    fig.suptitle('Acumulative Match Characteristic', fontsize=18, fontweight='bold')
    P.ylabel('%', fontsize=16)
    P.xlabel('Rank', fontsize=16)
    
    P.xlim(0,400)
    P.ylim(80,101)
    P.xticks(np.arange(0, 400, 10.0))
    P.yticks(np.arange(75, 101, 1.0))
    
    xticklabels = P.getp(P.gca(), 'xticklabels')
    yticklabels = P.getp(P.gca(), 'yticklabels')
    
    P.setp(yticklabels, 'color', 'k', fontsize='x-large')
    P.setp(xticklabels, 'color', 'k', fontsize='x-large')
    
    P.grid(True)
    fig.set_size_inches(19,7)
    #P.plot(cmc*100)
    P.plot(cmc*100)
    fig.savefig('cmc_bf_knn.png')
    P.show()

#%%%

#Author: Jacob Gildenblat, 2014
#http://jacobcv.blogspot.com.br/2014/12/fisher-vector-in-python.html
#License: you may use this for whatever you like
#Adaptation: Agnus A. Horta

def fv_dictionary(descriptors, N):
    
    import numpy as np
    import cv2
    
    em = cv2.ml.EM_create()
    em.setClustersNumber(N)
    #em = cv2.EM(N)
    em.trainEM(descriptors)

    return np.float32(em.getMeans()), \
        np.float32(em.getCovs()), np.float32(em.getWeights())[0]

def fv_generate_gmm(descriptors, N, dt):
    
    import numpy as np
    
    words = np.concatenate(descriptors)
    #np.concatenate([folder_descriptors(folder) for folder in glob.glob(input_folder + '*')])
    #print("Training GMM of size", N)
    means, covs, weights = fv_dictionary(words, N)
    #Throw away gaussians with weights that are too small:
    th = 1.0 / N
    means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
    covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
    weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

    #print 'Means: ',means
    #print 'Covs: ',covs
    #print 'Weights: ',weights
    
    np.save("./dat/means" + dt + ".gmm", means)
    np.save("./dat/covs" + dt + ".gmm", covs)
    np.save("./dat/weights" + dt + ".gmm", weights)
    
    return means, covs, weights

def fv_load_gmm(dt, folder = "./dat"):
    
    import numpy as np
    
    files = ["means" + dt + ".gmm" +".npy", "covs" + dt + ".gmm.npy", "weights" + dt + ".gmm.npy"]
    
    try:
        return map(lambda file: np.load(file), map(lambda s : folder + "/" + s , files))
        
    except IOError:
        return (None, None, None)
    
def fv_likelihood_moment(x, ytk, moment):
    import numpy as np
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk

def fv_likelihood_statistics(samples, means, covs, weights):

    from scipy.stats import multivariate_normal
    import numpy as np
    
    gaussians, s0, s1,s2 = {}, {}, {}, {}
    samples = zip(range(0, len(samples)), samples)

    #print samples

    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
    for index, x in samples:
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + fv_likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + fv_likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + fv_likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2

def fv_fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    import numpy as np    
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fv_fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    import numpy as np    
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fv_fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    import numpy as np
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def fv_normalize(fisher_vector):
    
    import numpy as np
    
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

def fv_fisher_vector(samples, means, covs, w):
    
    import numpy as np
    
    #print 'fisher_vector(samples, means, covs, w)'
    s0, s1, s2 =  fv_likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fv_fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fv_fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fv_fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    fv = fv_normalize(fv)
    
    #print 'fv = ', fv
    
    return fv

def le_descritores(sift_folder, subset, tipo=1):
    
    import os
    import numpy as np
    
    #n_folds = len(folds)
    #Alterei para que inclua nas imagens da galeria i no conj. train, de forma a que as
    # imagens correspondentes ao probe existam na galeria (train)
    #    for i in range(n_folds):
    #        train = folds[i][0]
    #        for j in range(n_folds):
    #            if j!=i :
    #                train = train + folds[j][0]+folds[j][1]+folds[j][2]
    #    
    #        n_train = len(train)

    ch = 0
    ds = []
    for image in subset:
        
        fname = os.path.join(sift_folder, image[:-3]+'sift_ds')
        ds1 = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.uint8) #,skiprows=1)
        
        if tipo = 1:
            if ch == 0:
                ch = 1                
                ds = []
                ds.append(ds1)
            else:
                ds.append(ds1)                
        else:
            if ch == 0:
                ch = 1
                ds = np.empty_like(ds1)
                ds[:] = ds1
            else:
                print ds.shape, ds1.shape
                ds = np.concatenate((ds, ds1), axis=0)
            
    return ds
 
#%%
def bov_histogramas_grava(arquivo, hists, dt):
    
    resultFile = open(arquivo, 'w')
    i = len(hists)
    for h in hists:
        line = (''.join(str(e) + ", " for e in h.tolist()))[:-2]
        resultFile.write(line)
        if i > 0:
            resultFile.write("\n")
        i = i - 1

    resultFile.close()

#%%

def bov_codebook_gera(l_sift, nc, tipo):

    if tipo == 1:

        # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit
        from sklearn.cluster import KMeans

        est = KMeans(n_clusters=nc, init='k-means++', n_init=10, max_iter=100,
                     tol=0.0001, precompute_distances='auto', verbose=0,
                     random_state=None, copy_x=True, n_jobs=4)
        est.fit(l_sift)
        labels = est.labels_
        centers = est.cluster_centers_

    elif tipo == 2:

        from sklearn.cluster import MiniBatchKMeans

        est = MiniBatchKMeans(n_clusters=nc, init='k-means++', max_iter=100,
                              batch_size=3*nc, verbose=0, compute_labels=True,
                              random_state=None, tol=0.0, max_no_improvement=10,
                              init_size=None, n_init=3, reassignment_ratio=0.01)
        est.fit(l_sift)
        labels = est.labels_
        centers = est.cluster_centers_

    else:

        import random
        from scipy.cluster.vq import vq
        import numpy as np

        list_of_random_items = random.sample(np.arange(l_sift.shape[0]), nc)
        l_centroids = []
        for i in list_of_random_items:
            l_centroids.append(l_sift[i])

        centers = np.asarray(l_centroids)
        labels, _ = vq(l_sift, centers)

    return (centers, labels)

#%%
def bov_histogramas_gera(labels, id_ds, indices, X, k, nomes_imagens, vis=False):

    from matplotlib import pyplot as plt
    import numpy as np

    fv = np.vectorize(f)
    
    hists = []
    i = 0

    for j in range(0, len(indices)):
        #ld = X[indices[j]].tolist()
        n = id_ds[j]
        sl = labels[i:i+n]

        hist, bins = np.histogram(sl, bins=k, range=(0, k), normed=False,
                                  weights=None, density=True)

        if vis == True:
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.title("Histogram "+nomes_imagens[indices[j]])
            plt.xlabel("Visual Word")
            plt.ylabel("Frequency")
            plt.bar(center, hist, align='center', width=width)
            plt.show()
            #print j

        hists.append(hist)
        #print hist
        i = i + n
        #j = j +1

    return hists

def bov_descritores_codifica(X, centers):
    from scipy.cluster.vq import vq
    
    labels,_ = vq(X,centers)

    return labels
 