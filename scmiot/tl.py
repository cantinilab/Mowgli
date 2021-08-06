# Biology
import scanpy as sc
import muon as mu
import anndata as ad
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt

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
