import numpy as np
import random
import networkx as nx
import time
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

# ==========================================
# Pure Numpy/NetworkX Implementation 
# (Removed Spektral and Tulip completely)
# ==========================================
def AM2Chebyshev(AM, K):
    N = AM.shape[0]
    I = np.eye(N)
    D = np.sum(AM, axis=1)
    
    D_inv_sqrt = np.power(np.maximum(D, 1e-12), -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    
    L = I - D_mat @ AM @ D_mat
    
    eigvals = np.linalg.eigvalsh(L)
    lambda_max = np.max(eigvals) if len(eigvals) > 0 else 2.0
    if lambda_max < 2.0:
        lambda_max = 2.0
        
    L_scaled = (2.0 / lambda_max) * L - I
    
    T = [I, L_scaled]
    for i in range(2, K + 1):
        T.append(2.0 * L_scaled @ T[i-1] - T[i-2])
        
    return np.stack(T[:K+1], axis=0)

def AM2GCNfilter(A, K):
    N = A.shape[0]
    I = np.eye(N)
    A_tilde = A + I
    D_tilde = np.sum(A_tilde, axis=1)
    
    D_inv_sqrt = np.power(np.maximum(D_tilde, 1e-12), -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    
    support = D_mat @ A_tilde @ D_mat
    return support.reshape(1, *support.shape)

def graph2AM(g, increased=-1):
    n = g.number_of_nodes()
    AM_size = max(n, int(increased))
    AM = np.zeros((AM_size, AM_size)) 
    for u, v in g.edges():
        AM[u][v] = 1
        AM[v][u] = 1
    return AM

def AM2DC(AM):
    degrees = np.sum(AM, axis=1)
    return degrees.reshape(-1, 1)

def AM2DM(AM, fill_value=0):
    DM = floyd_warshall(AM)
    if(type(fill_value) == int):
        DM[DM == np.inf] = fill_value
    elif(fill_value == "max"):
        tmp = -1
        DM[DM == np.inf] = tmp
        DM[DM == tmp] = np.max(DM)
    return DM

def AM2TH(AM, t=1.0):
    degree = np.sum(AM, axis=1)
    L = np.diag(degree) - AM
    eig_val, eig_vec = np.linalg.eigh(L)
    exp_eig_val = np.exp(-t * eig_val)
    H = eig_vec @ np.diag(exp_eig_val) @ eig_vec.T
    return H

def graphFeatures(g, features_names, increased=-1):
    n_nodes = max(g.number_of_nodes(), increased)
    fill_value = np.nan
    
    n_features = 0
    for feature in features_names:
        if feature.startswith("LAYOUT:"):
            n_features += 2
        else:
            n_features += 1
            
    F = np.full((n_nodes, n_features), fill_value)

    layouts = {}
    for feature in features_names:
        if feature.startswith("LAYOUT:"):
            algo = feature.split(":")[-1]
            if "Kamada" in algo:
                layouts[algo] = nx.kamada_kawai_layout(g)
            else:
                layouts[algo] = nx.random_layout(g)

    nodes = list(g.nodes())
    for i, n in enumerate(nodes):
        f_vec = []
        for feature in features_names:
            if feature.lower() == "id":
                f_vec.append(float(n))
            elif "random:" in feature:
                f_vec.append(random.uniform(0, 1))
            elif feature.startswith("LAYOUT:"):
                algo = feature.split(":")[-1]
                pos3d = layouts[algo][n]
                f_vec.append(float(pos3d[0])) 
                f_vec.append(float(pos3d[1]))             
        F[n] = f_vec
        
    for i in range(F.shape[1]):
        v = F[:, i]
        v[np.isnan(v)] = np.min(v[~np.isnan(v)])
    return F
    
def scaleFeatures(F, scalers):
    n_features = F.shape[-1]
    assert len(scalers) == n_features
    for f in range(n_features):
        fs = F[:,f].astype(float)
        fs = fs.flatten()
        fs = fs.reshape((*fs.shape, 1))            
        fs = scalers[f].transform(fs)
        F[:,f] = fs.reshape(F[:,f].shape)
    return F

def numpy_findSigma(DM, real_n, perplexity=30, n_iters=20, dtype="float32", epsilon=1e-16):
    DM = DM.astype(dtype)
    DM_red = DM[:real_n, :real_n]
    X = DM_red
    sigma = np.ones((real_n))
    target = np.log(perplexity)
    P = np.maximum(local_pij_cond_var(X, sigma), epsilon)
    entropy = -np.sum(P * np.log(P), axis=1)

    sigmin = np.full((real_n), epsilon, dtype=dtype)
    sigmax = np.full((real_n), np.inf, dtype=dtype)

    upmin = np.where(entropy < target, sigma, sigmin)
    upmax = np.where(entropy > target, sigma, sigmax)

    for i in range(n_iters):
        P = np.maximum(local_pij_cond_var(X, sigma), epsilon)
        entropy = -np.sum(P * np.log(P), axis=1)
        if np.any(np.isnan(np.exp(entropy))):
            return numpy_findSigma(DM, real_n, perplexity=perplexity*1.5, n_iters=n_iters, dtype=dtype, epsilon=epsilon)
        upmin = np.where(entropy < target, sigma, sigmin)
        upmax = np.where(entropy > target, sigma, sigmax)
        sigmin = upmin
        sigmax = upmax
        upsigma = np.where(np.isinf(sigmax), sigma*2, (sigmin + sigmax) / 2.)
        sigma = upsigma
    return sigma

def local_pij_cond_var(X, sigma):
    N = X.shape[0]
    sqdistance = X**2
    esqdistance = np.exp(-sqdistance / ((2 * (sigma**2)).reshape((N, 1))))
    np.fill_diagonal(esqdistance, 0)
    esqdistance_zd = esqdistance
    row_sum = np.sum(esqdistance_zd, axis=1).reshape((N, 1))
    return esqdistance_zd / row_sum  

def getSigma(DM, real_n, perplexity=None, sigma_iters=20):
    N = DM.shape[-1]
    if(perplexity is None):
        (Pmin, Pmax) = (5,N/2)
        (Nmin, Nmax) = (2, N)
        P = lambda n: (n-Nmin)*(Pmax-Pmin)/(Nmax-Nmin)+Pmin if Nmax>Nmin else Pmin
        perplexity = P(real_n)
    sigma = numpy_findSigma(DM, real_n, perplexity=perplexity, n_iters=sigma_iters)
    
    filled_sigma = np.full((N), np.sum(sigma))
    filled_sigma[:real_n] = sigma   
    filled_sigma = filled_sigma.reshape((N, 1))
    return filled_sigma 

def swapMatrix(M, real_n=None, permutation=None):
    assert len(M.shape) == 2 and M.shape[0] == M.shape[1]
    N = M.shape[0]
    if(permutation is not None):
        assert len(permutation) == N
    else:
        permutation = np.random.permutation(N)

    mask = np.zeros((N,N))
    if(real_n is not None):
        assert real_n <= N
        mask[:real_n, :real_n] = 1
    else:
        mask[:,:] = 1
        
    swappedM = np.zeros((N, N), np.float32)
    swappedMask = np.zeros((N, N), np.float32)
    for i in range(N):
        for j in range(N):
            swappedM[permutation[i]][permutation[j]] = M[i][j]
            swappedMask[permutation[i]][permutation[j]] = mask[i][j]
    return swappedM, swappedMask, permutation

def swapVector(V, real_n=None, permutation=None):
    assert len(V.shape) == 2
    N = V.shape[0]
    if(permutation is not None):
        assert len(permutation) == N
    else:
        permutation = np.random.permutation(N)

    mask = np.zeros((N,1))
    if(real_n is not None):
        assert real_n <= N
        mask[:real_n] = 1
    else:
        mask[:] = 1
        
    swappedV = np.zeros(V.shape)
    swappedMask = np.zeros(mask.shape)
    for i in range(N):
        swappedV[permutation[i]] = V[i]
        swappedMask[permutation[i]] = mask[i]
    return swappedV, swappedMask, permutation

def graph2predictData(g, graph_id, N_max, max_deg, model_inputs, features_names, topology_fn, swap=True, scalers=[]):
    AM = graph2AM(g, increased=N_max)
    DM = None
    perm = None
    real_n = g.number_of_nodes()
    N = max(N_max, real_n)
    input_dic = {}
    fake_inputs = True 
    if("DM" in model_inputs):
        if(fake_inputs):
            DM = np.zeros_like(AM)
        else:
            DM = AM2DM(AM)
        input_dic["DM"] = DM
    if("sigma" in model_inputs):
        if(fake_inputs):
            sigma = np.zeros((N, 1))
        else:
            if(DM is None):
                DM = AM2DM(AM)
            sigma =  getSigma(DM, real_n)
        input_dic["sigma"] =sigma
    if("DC" in model_inputs):
        real_AM = AM[:real_n, :real_n]
        real_DC = AM2DC(real_AM) 
        full_DC = np.zeros((N, 1))
        full_DC[:real_n] = real_DC
        input_dic["DC"] = full_DC
    if("TH" in model_inputs):
        if(fake_inputs):
            input_dic["TH"] = np.zeros_like(AM)
        else:
            real_AM = AM[:real_n, :real_n]
            real_TH = AM2TH(real_AM, t=1.0) 
            full_TH = np.zeros((N, N))
            full_TH[:real_n, :real_n] = real_TH
            input_dic["TH"] = full_TH

    start = time.time()
    if("nodesMask" in model_inputs):
        mask = np.zeros((N,1))
        mask[:real_n] = 1
        input_dic["nodesMask"] = mask
    if("features" in model_inputs):
        F = graphFeatures(g, features_names, increased=N_max)
        if(scalers is not None and len(scalers) > 0):
            F = scaleFeatures(F, scalers)
        input_dic["features"] = F
    if("supports" in model_inputs):
        cur_AM = AM
        if(swap):
            cur_AM, _, perm = swapMatrix(AM, real_n=real_n, permutation=perm)
        input_dic["supports"] = topology_fn(cur_AM, max_deg)    
    end = time.time()
    elapsed = end - start
    elapsed_ms = elapsed * 1000
    
    if(swap):
        swapperFn = {
            "DM":swapMatrix,
            "features":swapVector,
            "sigma":swapVector,
            "nodesMask":swapVector,
            "DC": swapVector,
            "TH": swapMatrix   
        }
        for input_name in model_inputs:
            if(input_name != "supports"):
                swapped_input, _, perm = swapperFn[input_name](input_dic[input_name], real_n=real_n, permutation=perm)
                input_dic[input_name] = swapped_input

    inputs = []
    for input_name in model_inputs:
        inputs.append(input_dic[input_name])
    return inputs, elapsed_ms