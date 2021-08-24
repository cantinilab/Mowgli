# Biology
import scanpy as sc
import muon as mu
import anndata as ad
from sklearn.metrics import r2_score, explained_variance_score, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

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

        for i in range(k):
            rec = mdata.mod[mod].uns['H_OT'][:,[i]] @ mdata.obsm['W_OT'][:,[i]].T
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

def leiden(mdata, n_neighbors=15, obsm='W_OT', resolution=1):
    joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_neighbors)
    sc.tl.leiden(joint_embedding, resolution=resolution)
    mdata.obs['leiden'] = joint_embedding.obs['leiden']

def inflexion_pt(a):
    second_derivative = [-np.inf]
    for i in range(1, len(a) - 1):
        second_derivative.append(a[i+1] + a[i-1] - 2 * a[i])
    return np.argmax(second_derivative)

def select_dimensions(mdata):
    latent_dim = mdata.obsm['W_OT'].shape[1]
    s = np.zeros(latent_dim)
    for mod in mdata.mod:
        s += np.array([(mdata[mod].uns['H_OT'][:,[k]] @ mdata.obsm['W_OT'].T[[k]]).std(1).sum() for k in range(latent_dim)])

    i = inflexion_pt(np.sort(s)[::-1])
    i = max(i, 5)
    plt.plot(np.sort(s)[::-1])
    plt.scatter(range(latent_dim), np.sort(s)[::-1])
    plt.scatter(range(i+1), np.sort(s)[::-1][:i+1])
    plt.plot(np.sort(s)[::-1][:i+1])
    plt.show()
    return np.argsort(s)[::-1][:i+1].copy()

def trim_dimensions(mdata, dims):
    mdata.obsm['W_OT'] = mdata.obsm['W_OT'][:,dims]

def sil_score(mdata, obsm='W_OT', obs='leiden'):
    return silhouette_score(mdata.obsm[obsm], mdata.obs[obs])

def best_leiden_resolution(mdata, obsm='W_OT', method='elbow', resolution_range=None, n_neighbors=15):
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

        second_derivative = [-np.inf]
        for i in range(1, len(vars)-1):
            second_derivative.append(vars[i+1] + vars[i-1] - 2 * vars[i])

        plt.xscale('log')
        plt.scatter(resolution_range, vars)
        plt.scatter(resolution_range[np.argmax(second_derivative)], vars[np.argmax(second_derivative)])
        plt.plot(resolution_range, vars)
        plt.ylabel('Average intra-cluster variation')
        plt.xlabel('Resolution')
        plt.show()

        return resolution_range[np.argmax(second_derivative)]

    elif method == 'silhouette':
        sils = []
        for res in resolution_range:
            sc.tl.leiden(joint_embedding, resolution=res)
            sils.append(silhouette_score(joint_embedding.X, joint_embedding.obs['leiden']))

        maxes = []
        for i in range(1, len(sils)-1):
            if sils[i] > sils[i+1] and sils[i] >= sils[i-1]:
                maxes.append(i)

        plt.xscale('log')
        plt.scatter(resolution_range, sils)
        plt.plot(resolution_range, sils)
        plt.scatter([resolution_range[maxes[0]]], [sils[maxes[0]]])
        plt.ylabel('Silhouette score')
        plt.xlabel('Resolution')
        plt.show()

        return resolution_range[maxes[0]]
