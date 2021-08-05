# Single-cell multi-omics integration with Optimal Transport

**Install package**

    git clone git@github.com:gjhuizing/OT-scIntegration.git
    pip install ./OT-scIntegration/

**Import package**: run `from scmiot import models, pl`

**Conda environment**: `environment.yml` (in particular, relies on pytorch, scanpy and muon)

**Datasets**: since the data is too big to be hosted on github, please download and unzip the file `datasets.zip` available at https://hub.bio.ens.psl.eu/index.php/s/wyKyyTPTXAww4nQ/download

**Vignettes**: separate files for preprocessing, and application of OT-NMF

**Tool**: in folder `scmiot/`, `models.py` contains models, and `pl.py` plotting functions
