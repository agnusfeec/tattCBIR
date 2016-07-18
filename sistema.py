# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:20:08 2016

@author: agnus
"""
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

def grava_db_imagens(arquivo, imagens):
    #arquivo = './tatt_c.db'
    with open(arquivo, 'wb') as db_image_file:
        for  nome_img, caminho in imagens.items():
            db_image_file.write(nome_img+ '\t' + caminho + '\n')
        db_image_file.close()

def grava_config(arquivo = './example.cfg'):

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
    config.set('Geral', 'Database Image Folder', '/media/sf_Projeto/dataset/tatt-c/')
    config.set('Geral', 'Indexa image database', 'True')
    config.set('Geral', 'Database filename', './tatt_c.db')
    config.set('Geral', 'Image filename extension','.jpg')

    config.set('Geral', 'Training File', 'train1')
    config.set('Geral', 'Testing File', 'test1')

    config.add_section('Folds')
    config.set('Folds', 'Folds Folder', '/media/sf_Projeto/dataset/tatt-c_update_v1.4/5-fold/tattoo_identification/')
    config.set('Folds', 'Quantidade subsets', '2')
    config.set('Folds', 'Subset_1', 'gallery{1,2,3,4,5}.txt')
    config.set('Folds', 'Subset_2', 'probes{1,2,3,4,5}.txt')
    config.set('Folds', 'Ground_truth', 'ground_truth.txt')
    config.add_section('SIFT')
    config.set('SIFT','SIFT Folder',  '/media/sf_Projeto/dataset/tatt-c_SIFT/')

    # Writing our configuration file to 'example.cfg'
    with open(arquivo, 'wb') as configfile:
        config.write(configfile)


def le_config():

    import ConfigParser

    config = ConfigParser.RawConfigParser()
    config.read('./example.cfg')

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
        aux = []
        for j in range(n_subsets):
            arquivo = subsets[j][i]
            with open(folds_folder+arquivo, 'r') as imagefiles:
                for nomef in imagefiles:
                    if nomef[-1] == '\n' : nomef = nomef[:-1]
                    aux.append(nomef)
        folds.append(aux)

    #print folds[0]

    gt_filename = config.get('Folds', 'ground_truth')
    gt_imagens = {}
    with open(folds_folder+gt_filename, 'r') as gt_arq:
        for nomef in gt_arq:
            imgs = nomef.split('|')
            if imgs[1][-1] == '\n' : imgs[1] = imgs[1][:-1]
            #print imgs[0], imgs[1]
            gt_imagens[imgs[0]] = imgs[1]
        gt_arq.close()

    sift_folder = config.get('SIFT', 'sift folder')

    return folds, imagens, gt_imagens, sift_folder



#dbImage_folder ="/home/agnus/Documentos/leo/datasets/tatt-c/"
#imagens = monta_lista_imagens(dbImage_folder,".jpg")
#print imagens['gallery_011.jpg']
#print imagens['group10_img01.jpg']

#arquivo = './tatt_c.db'
#with open(arquivo, 'wb') as db_image_file:
#    for  nome_img, caminho in imagens.items():
#        db_image_file.write(nome_img+ '\t' + caminho + '\n')
#db_image_file.close()

grava_config()

def sift(nomes_imagens, imagens, sift_folder):

    import cv2
    import os

    #ds = []
    #kp = []
    t = len(nomes_imagens)
    i=1


    for filename in nomes_imagens:

        fname = os.path.join(sift_folder, filename[:-3]+'sift')

        if os.path.isfile(fname) == False :
            print filename
            #file_img = os.path.join(diretorio, filename)
            diretorio = imagens[filename]
            img = cv2.imread(os.path.join(diretorio, filename)) #file_img)
            # Redimensiona imagem para aplicação do Fisher Vectors
            #img = cv2.resize(img, (256,256))
            aux = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(aux)
            sift = cv2.xfeatures2d.SIFT_create()
            (kps, descs) = sift.detectAndCompute(gray, None)

            #ds.append(descs)
            #kp.append(kps)

            arquivo = os.path.join(sift_folder, filename[:-3]+'sift')
            with open(arquivo, 'wb') as sift_file:
                for desc in descs:
                    sift_file.write(','.join(str(x) for x in desc)+'\n')
                sift_file.close()

        print (i*100)/t,
        i=i+1

    #return ds

def sift_match(ds1, ds2):

    import cv2

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ds1,ds2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    qm = len(good)


    (nr1,c) = ds1.shape
    (nr2,c) = ds2.shape

    nr = nr1
    if nr2>nr:
        nr = nr2

    #rt = (100.0*qm/nr)
    if qm > 0:
        rt = 1.0/qm
    else:
        rt = 10^8

    return rt

folds, imagens, gt_images, sift_folder = le_config()
n_folds = len(folds)

#%%
import numpy as np
import os

# Inicialmente gera se necessario o SIFT para as imagens de treinamento e teste
# pode ser otimizado, gerando para toda a base, caso se utilize toda a base
# o que pode ter um custo alto pois na base existem imagens para outros casos
# de uso.

for i in range(n_folds):
    test = folds[i]
    train = []
    for j in range(n_folds):
        if j!=i :
            train = train + folds[j]

    print 'Gerando sift do conjunto de treinamento'
    #train_kp, train_ds = sift(train, imagens, sift_folder)
    sift(train, imagens, sift_folder)
    print 'Gerando sift do conjunto de teste'
    #test_kp, test_ds = sift(test, imagens)
    sift(test, imagens, sift_folder)
    
for i in range(n_folds):
    test = folds[i]
    train = []
    for j in range(n_folds):
        if j!=i :
            train = train + folds[j]

    n_test = len(test)
    n_train = len(train)

    dist = np.zeros((1, n_train), dtype=np.float)
    nn = n_test * n_train
    i = 0

    print 'Gerando o match entre o treinamento e o conjunto de teste'

    arquivo = './clist.txt'
    with open(arquivo, 'w') as clist_file:

        for file_test in test:
    
            fname = os.path.join(sift_folder, file_test[:-3]+'sift')
            ds1 = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.uint8) #,skiprows=1)
            j = 0
    
            for file_train in train:
                fname = os.path.join(sift_folder, file_train[:-3]+'sift')
                ds2 = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.uint8) #,skiprows=1)
                #print ds1
                #print ds2
                rt = sift_match(ds1,ds2)
                dist[0][j] = rt
                j = j + 1
                print (((i*n_train)+j)*100)/nn,
    
            indice = (np.argsort(dist)).tolist()
            k = 1
            for id in indice[0]:
                clist_file.write(file_test+'|'+ str(k) + '|' + train[id] + '|' + str(dist[0][id]) +'\n')
                k = k + 1
                
            i = i + 1

        clist_file.close()

#%%

#for filename in test:
#    print filename, imagens[filename]
