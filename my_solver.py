import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
# from sklearn.externals import joblib
import joblib
from numpy import genfromtxt
from pandas import DataFrame
from sklearn.impute import SimpleImputer as Imputer
from sklearn import preprocessing


# loss function
def loss(w, nl,d, h, Th, X, v, Y, lambd):
    w = w[:, np.newaxis]
    lossV=(1.0/nl)*np.power(np.linalg.norm(np.dot(np.transpose(X),w)+np.dot(np.transpose(X),np.dot(np.transpose(Th),np.transpose(v))) - Y),2)+ lambd*np.power(np.linalg.norm(w),2)
   # print (lossV)
    return lossV

# gradient of loss
def grad(w, nl,d, h, Th, X, v, Y, lambd):
    w=w[:,np.newaxis]
    return (1.0/nl)*(2* np.dot(X, np.dot(np.transpose(X),w)+np.dot(np.transpose(X),np.dot(np.transpose(Th),np.transpose(v))) -Y)) +2*lambd*w
    

#  read file    
def read_split_aug(filepath, filename, rmv, finalNames):
    #read the csv
    try:
        dataset = genfromtxt(open(filepath + '/' + filename,'rb'), delimiter=',', dtype='f8')[0:]
            
        # Clean the dataset
        # Sort the observations according to the timestamp
        dataset = dataset[dataset[:,0].argsort()]
        dataset = dataset[12:, :] #exclude some nan observations
        dataset = dataset[0:360, :] #exclude some nan observations
    
        # Remove redundant resources  
        dataset = np.delete(dataset, np.s_[rmv], axis = 1)
        target = dataset[:,1865]     #values of the target variable
        tt = target[:,np.newaxis]
        rm_dataset = np.delete(dataset,np.s_[2453:2455], axis = 1)      #exclude VOD
        rm_dataset = np.delete(rm_dataset, np.s_[1863:1869], axis=1)    #exclude NDVI 
        rm_dataset = np.delete(rm_dataset, np.s_[1849:1853], axis=1)    #exclude VOD
        dataset =np.concatenate((rm_dataset, tt), axis=1) #put the target column in the end
        dataset = DataFrame(dataset)
        
        dataset = dataset.fillna(0)
        dataset.columns = finalNames.ravel()
        names = dataset.columns[3:dataset.shape[1]]
        
        # Creat the new dataset 
        X = dataset.iloc[:,3:dataset.shape[1]-1]
        y = dataset.iloc[:,dataset.shape[1]-1]
        #  import the lags of NDVI (target)
        win=13
        new_datasetAuto = np.empty((len(y),win))
        for i in range(1,win):
            new_datasetAuto[:,i-1] = shift2(y, i)#, cval=np.NaN)
        new_datasetAuto[:,win-1] = y
        
        # Imputer the missing values with the mean 
        # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp = Imputer(missing_values=np.nan, strategy='mean')# fill in the missing values with the mean of each column, works on axis=0 by default
        dataImputedAuto = imp.fit_transform(new_datasetAuto)       
        X1 = dataImputedAuto[:,0:dataImputedAuto.shape[1]-1]
        X = np.concatenate((X, X1), axis=1)
        new_dataset = np.concatenate((X, DataFrame(y)), axis=1)
        new_dataset = DataFrame(new_dataset)
        
        # Imputer the missing values with zero 
        new_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_dataset = new_dataset.fillna(0)
        
        predictor_names = names[0:len(names)-1].tolist()
        target_name = names[len(names)-1]
        for i in range(1,13):
            predictor_names.append(target_name + str(i))
        predictor_names.append(target_name)
        predictor_names= np.array(predictor_names)
        new_dataset.columns = predictor_names.ravel()
    
        return new_dataset
    except IOError as e:#if the file does not exist throw an exception
        #print e
        return []
        pass

# shifts a given time series num times
def shift2(arr,num):
    arr=np.roll(arr,num)
    if num<0:
         np.put(arr,range(len(arr)+num,len(arr)),np.nan)
    elif num > 0:
         np.put(arr,range(num),np.nan)
    return arr

# returns informative features (columns of the dataset)
def feature_indices(names):
    t_era = []
    t_isccp = []
    t_mlost = []
    t_giss = []
    t_udel = []
    t_cru = []
    t_lst = []
    rn_era = []
    rn_srb = []
    sm_comb = []
    sm_pass = []
    sm_gleam = []
    swe = []
    p_era = []
    p_udel = []
    p_gpcc = []
    p_gpcp = []
    p_cpcu = []
    p_cru = []
    p_cmap = []
    p_mswep = []
    rest = []
    ndvi = []
    RNcum7 = []
    RNlag7 = []
    Wcum7 = []
    Wlag7 = []
    Tcum7 = []
    Tlag7 = []
    for i, s in enumerate(names):
        digit=100
        if ('T_' in s) & ~('P_3B' in s):
            if 'ERA' in s:
                t_era.append(i)
            elif 'ISCCP' in s:
                t_isccp.append(i)
            elif 'MLOST' in s:
                t_mlost.append(i)
            elif 'GISS' in s:
                t_giss.append(i)
            elif 'UDEL' in s:
                t_udel.append(i)
            elif 'CRU' in s:
                t_cru.append(i)
            elif 'LST' in s:
                t_lst.append(i)
            else:
                rest.append(i)
            if ('_cum' in s):
                varToks =str.split(s,'_')
                for dig in varToks:
                    digitFound = dig.isdigit()
                    if(digitFound):
                        digit = float(dig)
                    if(digitFound & (digit<10)):
                        cum = float(dig)
                        break
                if cum >= 7:
                   Tcum7.append(i)
            elif ('Lag' in s):
                varToks =str.split(s,'_')
                lag = float(varToks[-1])
                if lag >= 7:
                   Tlag7.append(i)
        elif 'RN_' in s:
            if 'ERA' in s:
                 rn_era.append(i)
            else:
                 rn_srb.append(i)
            if ('_cum' in s):
                varToks =str.split(s,'_')
                for dig in varToks:
                    digitFound = dig.isdigit()
                    if(digitFound):
                        digit = float(dig)
                    if(digitFound & (digit<10)):
                        cum = float(dig)
                        break
                if cum >= 7:
                   RNcum7.append(i)
            elif ('Lag' in s):
                varToks =str.split(s,'_')
                lag = float(varToks[-1])
                if lag >= 7:
                   RNlag7.append(i) 
        elif (('SM_' in s) & ~('ISCCP_' in s)):
            if 'COMBINED' in s:
                 sm_comb.append(i)
            elif 'PASSIVE' in s:
                 sm_pass.append(i)
            elif 'GLEAM' in s:
                 sm_gleam.append(i)
            else:
                 rest.append(i)
            if ('_cum' in s):
                varToks =str.split(s,'_')
                if 'CDSD' in s:
                    varToks = varToks[2:len(varToks)]
                for dig in varToks:
                    digitFound = dig.isdigit()
                    if(digitFound):
                        digit = float(dig)
                    if(digitFound & (digit<=12)):
                        cum = float(dig)
                        break
                if cum >= 7:
                   Wcum7.append(i) 
            elif ('Lag' in s):
                varToks =str.split(s,'_')
                lag = float(varToks[-1])
                if lag >= 7:
                   Wlag7.append(i) 
        elif ('GLOBSNOW' in s):
             swe.append(i)
             if ('_cum' in s):
                varToks =str.split(s,'_')
                if 'CDSD' in s:
                    varToks = varToks[2:len(varToks)]
                for dig in varToks:
                    digitFound = dig.isdigit()
                    if(digitFound):
                        digit = float(dig)
                    if(digitFound & (digit<=12)):
                        cum = float(dig)
                        break

                if cum >= 7:
                   Wcum7.append(i) 
             elif ('Lag' in s):
                varToks =str.split(s,'_')
                lag = float(varToks[-1])
                if lag >= 7:
                   Wlag7.append(i) 
        elif ( ('P_' in s) & ~('ISCCP_' in s)):
             if 'CPCU' in s:
                 p_cpcu.append(i)
             elif 'ERA' in s:
                 p_era.append(i)
             elif 'GPCC' in s:
                 p_gpcc.append(i)
             elif 'UDEL' in s:
                 p_udel.append(i)
             elif 'CRU' in s:
                 p_cru.append(i)
             elif 'CMAP' in s:
                 p_cmap.append(i)
             elif 'GPCP' in s:
                 p_gpcp.append(i)
             elif 'MSWEP' in s:
                 p_mswep.append(i)
             else:
                rest.append(i)
             if ('_cum' in s):
                varToks =str.split(s,'_')
                if 'CDSD' in s:
                    varToks = varToks[2:len(varToks)]
                for dig in varToks:
                    digitFound = dig.isdigit()
                    if(digitFound):
                        digit = float(dig)
                    if(digitFound & (digit<=12)):
                        cum = float(dig)
                        break
                if cum >= 7:
                   Wcum7.append(i) 
             elif ('Lag' in s):
                varToks =str.split(s,'_')
                lag = float(varToks[-1])
                if lag >= 7:
                   Wlag7.append(i) 
        elif ('NDVI' in s):
             ndvi.append(i)
        else:
             rest.append(i)
    
    #find the indices for each variable
    rad = rn_era + rn_srb
    wat = sm_comb + sm_pass + sm_gleam + swe+p_era + p_udel + p_gpcc + p_gpcp + p_cpcu + p_cru + p_cmap + p_mswep
    temp = t_era + t_isccp + t_mlost + t_giss + t_udel + t_cru + t_lst
    #exclude lags and cums >6
    wat=[x for x in wat if x not in Wlag7] 
    wat=[x for x in wat if x not in Wcum7] 
    temp=[x for x in temp if x not in Tlag7]
    temp=[x for x in temp if x not in Tcum7]       
    
    return wat, temp, rad, ndvi
    
# keep the selected columns 
def permutation_data(dataset, permutation):
    perm_data = dataset.iloc[:,permutation]
    return perm_data
    
# load the dataset (.csv file for each location)
def load_dataset(inpath, filename, varfile, remove_ind_file):
    mfile = varfile # load file with variable names
    name_vector = ["values"]
    varNames = pd.read_csv(mfile, names=name_vector)
    varNames = varNames.values

    # Delete redundant resources
    rmv = joblib.load(remove_ind_file)#load file with redundant resources, columns not-used in the analysis
    varNames = np.delete(varNames, np.s_[rmv], axis = 0)

    # Delete redundant vars
    targetVar = varNames[1865,:] #this is the name of the target variable (NDVI_GIMMS_Residuals)
    rm_varNames = np.delete(varNames, np.s_[2453:2455], axis = 0)    #exclude VOD
    rm_varNames = np.delete(rm_varNames, np.s_[1863:1869], axis=0)   #exclude NDVI
    rm_varNames = np.delete(rm_varNames, np.s_[1849:1853], axis=0)   #exclude VOD
    targetVarNew = targetVar[:,np.newaxis] 
    finalNames = np.concatenate((rm_varNames, targetVarNew), axis=0) #put the name of the target in the end
    dataset1 = read_split_aug(inpath, filename, rmv, finalNames)
    names = dataset1.columns.ravel()
    names = names.tolist()
    wat, temp, rad, ndvi = feature_indices(names)
    permutation = wat+temp+rad+ndvi
    temp_dataset = permutation_data(dataset1, permutation)
    min_max_scaler = preprocessing.StandardScaler()
    Xdata_norm = min_max_scaler.fit_transform(temp_dataset.iloc[:,:-1]) # keep only the features X
    y_data = temp_dataset.iloc[:,-1] # keep only the target y
    return Xdata_norm, y_data
    
    
# initialization    
def initialize_vars(X,y, W, Th, U, lambdas, task, h=100):

    X = np.transpose(X)
    Y = y[:,np.newaxis]
    nl = X.shape[1]
    d = X.shape[0]
    h = h
    
    u = U[:,task]
    u = u[:,np.newaxis]
    v = np.dot(Th,u)
    v = np.transpose(v)
    lambd = lambdas[0,task]
    W0 = W[:,task]
    W0 = W0[:,np.newaxis]
    return X, Y, Th, u, v, lambd, W0, nl, d, h

# solves an optimization problem for each location  
def solver(inpath, filename, W, Th, U, lambdas, task, h, varfile, remove_ind_file):
    lat,lon = filename
    filename = '%s,%s.csv'%(lat,lon)
    xdat, ydat = load_dataset(inpath, filename, varfile, remove_ind_file)
    X, Y, Th, u, v, lambd, W0, nl, d, h  = initialize_vars(xdat,ydat, W, Th, U, lambdas, task, h)
    res=fmin_l_bfgs_b(loss, W0, fprime=grad, args=(nl,d, h, Th, X, v, Y, lambd),iprint = -1,maxfun=30000, maxiter=30000)
    return res, Th, v, lambd