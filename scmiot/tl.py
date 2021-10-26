# Biology
import scanpy as sc
import muon as mu
import anndata as ad
from sklearn.metrics import r2_score, explained_variance_score, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import pearsonr, spearmanr

def umap(mdata: mu.MuData, obsm: str, n_neighbors: int = 15, metric: str = 'euclidean') -> None:
    """Compute UMAP of the given `obsm`.

    Args:
        mdata (mu.MuData): Input data
        obsm (str): The embedding
        n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 15.
        metric (str, optional): Which metric to compute neighbors. Defaults to 'euclidean'.
    """
    joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    sc.pp.neighbors(joint_embedding, n_neighbors=n_neighbors, metric=metric)
    sc.tl.umap(joint_embedding)

    mdata.obsm[obsm + '_umap'] = joint_embedding.obsm['X_umap']

def sil_score(mdata: mu.MuData, obsm: str, obs: str) -> float:
    """Compute silhouette score of an embedding

    Args:
        mdata (mu.MuData): Input data
        obsm (str): Embedding
        obs (str): Annotation

    Returns:
        float: Silhouette score
    """
    return silhouette_score(mdata.obsm[obsm], mdata.obs[obs])



def variance_explained(mdata, score_function='explained_variance_score', plot=True):
    """experimental, i have to test this function"""
    if score_function == 'explained_variance_score':
        f_score = explained_variance_score
    elif score_function == 'r2_score':
        f_score = r2_score
    else:
        f_score = explained_variance_score
        print('function not recognized, defaulting to explained_variance_score')
    score = []
    k = mdata.obsm['W_OT'].shape[1]
    for mod in mdata.mod:
        score.append([])
        A = mdata.mod[mod].uns['H_OT'] @ mdata.obsm['W_OT'].T
        A = A.cpu().numpy()

        for i in range(k):
            rec = mdata.mod[mod].uns['H_OT'][:,[i]] @ mdata.obsm['W_OT'][:,[i]].T
            rec = rec.cpu().numpy()
            score[-1].append(f_score(A, rec))

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(score, aspect='auto', interpolation='none')
        ax.set_xticks(range(k))
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Modality')
        ax.set_yticks(range(len(mdata.mod)))
        ax.set_title('Variance explained')
        ax.set_yticklabels(mdata.mod.keys())
        plt.colorbar()
        plt.show()
    return score

def leiden_multi(mdata, n_neighbors=15, obsm='W_OT', obs='rna:celltype', resolutions=10.**np.linspace(-2, 1, 20)):
    try:
        joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    except:
        joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    aris = []
    nmis = []
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)
    for resolution in tqdm(resolutions):
        sc.tl.leiden(joint_embedding, resolution=resolution)
        aris.append(ARI(joint_embedding.obs['leiden'], mdata.obs[obs]))
        nmis.append(NMI(joint_embedding.obs['leiden'], mdata.obs[obs]))
    return resolutions, aris, nmis

def leiden_multi_obsp(mdata, n_neighbors=15, neighbors_key='wnn', obs='rna:celltype', resolutions=10.**np.linspace(-2, 1, 20)):
    aris = []
    nmis = []
    for resolution in tqdm(resolutions):
        sc.tl.leiden(mdata, resolution=resolution, neighbors_key=neighbors_key, key_added='leiden_' + neighbors_key)
        aris.append(ARI(mdata.obs['leiden_' + neighbors_key], mdata.obs[obs]))
        nmis.append(NMI(mdata.obs['leiden_' + neighbors_key], mdata.obs[obs]))
    return resolutions, aris, nmis

def leiden(mdata, n_neighbors=15, obsm='W_OT', resolution=1):
    try:
        joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    except:
        joint_embedding = ad.AnnData(mdata.obsm[obsm], obs=mdata.obs)
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)
    sc.tl.leiden(joint_embedding, resolution=resolution)
    mdata.obs['leiden'] = joint_embedding.obs['leiden']

def inflexion_pt(a):
    second_derivative = [-np.inf]
    for i in range(1, len(a) - 1):
        second_derivative.append(a[i+1] + a[i-1] - 2 * a[i])
    return np.argmax(second_derivative)

def select_dimensions(mdata, plot=True):
    latent_dim = mdata.obsm['W_OT'].shape[1]
    s = np.zeros(latent_dim)
    for mod in mdata.mod:
        s += np.array([(mdata[mod].uns['H_OT'][:,[k]] @ mdata.obsm['W_OT'].T[[k]]).std(1).sum() for k in range(latent_dim)])

    i = inflexion_pt(np.sort(s)[::-1])
    i = max(i, 4)
    if plot:
        plt.plot(np.sort(s)[::-1])
        plt.scatter(range(latent_dim), np.sort(s)[::-1])
        plt.scatter(range(i+1), np.sort(s)[::-1][:i+1])
        plt.plot(np.sort(s)[::-1][:i+1])
        plt.show()
    return np.argsort(s)[::-1][:i+1].copy()

def trim_dimensions(mdata, dims):
    mdata.obsm['W_OT'] = mdata.obsm['W_OT'][:,dims]
    for mod in mdata.mod:
        mdata[mod].uns['H_OT'] = mdata[mod].uns['H_OT'][:,dims]

def predict_features_corr(mdata, mod, n_neighbors, features_idx, remove_zeros=True, obsp=None, obsm=None):
    if obsp:
        distances = np.array(mdata.obsp[obsp].todense())
        distances[distances == 0] = np.max(distances)
        np.fill_diagonal(distances, 0)
    else:
        distances = cdist(mdata.obsm[obsm], mdata.obsm[obsm])
    pred = np.zeros((mdata.n_obs, len(features_idx)))
    for i in tqdm(range(mdata.n_obs)):
        idx = distances[i].argsort()[1:1+n_neighbors]
        pred[i] = np.mean(mdata[mod].X[idx][:,features_idx], axis=0)
    truth = np.array(mdata[mod].X[:, features_idx])

    pearson, spearman = [], []
    for i in range(len(features_idx)):
        x = truth[:,i]
        y = pred[:,i]
        idx = x > 0 if remove_zeros else np.arange(len(x))
        pearson.append(pearsonr(x[idx], y[idx])[0])
        spearman.append(spearmanr(x[idx], y[idx])[0])
    
    return pearson, spearman

def knn_predict(mdata, obs, obsm='W_OT', max_neighbors=15):
    distances = cdist(mdata.obsm[obsm], mdata.obsm[obsm])
    s = 0
    for i in tqdm(range(mdata.n_obs)):
        idx = distances[i].argsort()[1:max_neighbors]
        s += np.cumsum(np.array(mdata.obs[obs][i] == mdata.obs[obs][idx]))/np.arange(1, max_neighbors)
    return s / mdata.n_obs

def knn_score(mdata, obs, obsm='W_OT', max_neighbors=15):
    distances = cdist(mdata.obsm[obsm], mdata.obsm[obsm])
    s = 0
    for i in tqdm(range(mdata.n_obs)):
        idx = distances[i].argsort()[1:max_neighbors]
        s += np.cumsum(np.array(mdata.obs[obs][i] == mdata.obs[obs][idx]))/np.arange(1, max_neighbors)
    return s / mdata.n_obs

def best_leiden_resolution(mdata, obsm='W_OT', method='elbow', resolution_range=None, n_neighbors=15, plot=True):
    if resolution_range==None:
        resolution_range = 10.**np.linspace(-2, 1, 20)

    if method != 'elbow' and method != 'silhouette':
        print('method not recognized, defaulting to elbow')
        method = 'elbow'

    joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)

    if method == 'elbow':
        vars = []
        for res in resolution_range:
            sc.tl.leiden(joint_embedding, resolution=res)
            wss = []
            for cat in joint_embedding.obs['leiden'].unique():
                wss.append(joint_embedding.X[joint_embedding.obs['leiden'] == cat].std(0).sum())
            vars.append(np.mean(wss))

        i = inflexion_pt(vars)
        if plot:
            plt.xscale('log')
            plt.scatter(resolution_range, vars)
            plt.scatter(resolution_range[i], vars[i])
            plt.plot(resolution_range, vars)
            plt.ylabel('Average intra-cluster variation')
            plt.xlabel('Resolution')
            plt.show()

        return resolution_range[i]

    elif method == 'silhouette':
        sils = []
        for res in resolution_range:
            sc.tl.leiden(joint_embedding, resolution=res)
            try:
                sils.append(silhouette_score(joint_embedding.X, joint_embedding.obs['leiden']))
            except:
                sils.append(-1)

        maxes = []
        for i in range(1, len(sils)-1):
            if sils[i] > sils[i+1] and sils[i] >= sils[i-1]:
                maxes.append(i)

        if plot:
            plt.xscale('log')
            plt.scatter(resolution_range, sils)
            plt.plot(resolution_range, sils)
            plt.scatter([resolution_range[maxes[0]]], [sils[maxes[0]]])
            plt.ylabel('Silhouette score')
            plt.xlabel('Resolution')
            plt.show()

        return resolution_range[maxes[0]]
