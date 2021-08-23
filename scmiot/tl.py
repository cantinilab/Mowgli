# Biology
import scanpy as sc
import muon as mu
import anndata as ad
from sklearn.metrics import r2_score, explained_variance_score, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def variance_explained(mdata, score_function='explained_variance_score'):
    print('experimental, i have to test this function')
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

def leiden(mdata, n_components=None, obsm='W_OT', resolution=1):
    joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    if n_components==None:
        n_components = joint_embedding.n_vars
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_components)
    sc.tl.leiden(joint_embedding, resolution=resolution)
    mdata.obs['leiden'] = joint_embedding.obs['leiden']

def sil_score(mdata, obsm='W_OT', obs='leiden'):
    return silhouette_score(mdata.obsm[obsm], mdata.obs[obs])

def best_leiden_resolution(mdata, obsm='W_OT', resolution_range=None, return_second_max=True, n_components=None):
    if resolution_range==None:
        resolution_range = 10.**np.linspace(-2, 1, 20)
    sils = []

    joint_embedding = ad.AnnData(mdata.obsm[obsm].cpu().numpy(), obs=mdata.obs)
    if n_components==None:
        n_components = joint_embedding.n_vars
    sc.pp.neighbors(joint_embedding, use_rep="X", n_neighbors=n_components)

    for res in resolution_range:
        sc.tl.leiden(joint_embedding, resolution=res)
        sils.append(silhouette_score(joint_embedding.X, joint_embedding.obs['leiden']))
    plt.xscale('log')
    plt.scatter(resolution_range, sils)
    plt.plot(resolution_range, sils)
    plt.ylabel('Silhouette score')
    plt.xlabel('Resolution')
    plt.show()

    maxes = []
    for i in range(1, len(sils)-1):
        if sils[i] > sils[i+1] and sils[i] >= sils[i-1]:
            maxes.append(i)
    if return_second_max:
        return resolution_range[maxes[1]]
    else:
        return resolution_range[maxes]
