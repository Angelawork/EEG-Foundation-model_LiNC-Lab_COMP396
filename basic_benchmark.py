import warnings
import mne
import moabb
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb import benchmark
import os
import random
from utils import *
from dataset_setup import *
from pipeline import *

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM,FgMDM
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from keras.losses import BinaryCrossentropy
from moabb.pipelines.deep_learning import KerasEEGNeX, KerasEEGTCNet, KerasEEGNet_8_2, KerasEEGITNet, KerasShallowConvNet
from moabb.pipelines import ExtendedSSVEPSignal
from moabb.pipelines.csp import TRCSP
from moabb.pipelines.features import AugmentedDataset
from pyriemann.spatialfilters import CSP
#converts matrix 3D > 2D feature vector.
from moabb.pipelines.features import LogVariance
from moabb.pipelines.utils import parse_pipelines_from_directory

moabb.set_log_level("info")
mne.set_log_level("CRITICAL")
warnings.filterwarnings("ignore")

def set_seed(s):
    np.random.seed(s)
    random.seed(s)
    tf.random.set_seed(s)

SCRATCH = os.environ["SCRATCH"]
SLURM_TMPDIR = os.environ["SLURM_TMPDIR"]
from mne import get_config
from moabb.utils import set_download_dir

original_path = get_config("MNE_DATA")
print(f"The download directory is currently {original_path}")
new_path = SCRATCH+"/mne_data"
set_download_dir(new_path)
check_path = get_config("MNE_DATA")
print(f"Now the download directory has been changed to {check_path}")

os.environ["MNE_DATA"] = check_path
os.environ["MOABB_RESULTS"] = check_path
print(f"Path for MNE_DATA: {os.environ['MNE_DATA']}")

#----------------------------pipeline definition----------------------------
pipelines = {}
param_grid = {}
# ACM+TS+SVM
# pipelines["ACM+TS+SVM"] = make_pipeline(
#     AugmentedDataset(),Covariances("cov"), TangentSpace(metric="riemann"), SVC(kernel="rbf")
# )

# TS+LR
pipelines["TS+LR"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression(C=1.0)
)

# FgMDM
pipelines["FgMDM"] = make_pipeline(
    Covariances("oas"), FgMDM(metric="riemann")
)

# TS+SVM
# pipelines["TS+SVM"] = make_pipeline(
#     Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear")
# )
pipelines["TS+SVM"] = Pipeline(
    steps=[("Covariances", Covariances("oas")),
        ("Tangent_Space", TangentSpace(metric="riemann")),
        (
            "svc",
            SVC(kernel="linear"),
        ),
    ]
)
param_grid["TS+SVM"] = {
    "svc__C": [0.5,1,1.5],
    "svc__kernel":["rbf","linear"],
}

# TS+EL
pipelines["TS+EL"] = Pipeline(
    steps=[("Covariances", Covariances("oas")),
        ("Tangent_Space", TangentSpace(metric="riemann")),
        (
            "logisticregression",
            LogisticRegression(
            penalty="elasticnet",       
            l1_ratio=0.70,             
            intercept_scaling=1000.0,   
            solver="saga",               
            max_iter=1000),
        ),
    ]
)
param_grid["TS+EL"] = {
    "logisticregression__l1_ratio": [0.20,0.30,0.45,0.65,0.75]
}

# # FilterBank+SVM
# pipelines["FilterBank+SVM"] = make_pipeline(
#     ExtendedSSVEPSignal(), Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear")
# )

# CSP+SVM
pipelines["CSP+SVM"] = Pipeline(
    steps=[("Covariances", Covariances("oas")),
        ("csp", CSP(nfilter=6)),
        (
            "svc",
            SVC(kernel="linear"),
        ),
    ]
)
param_grid["CSP+SVM"] = {
    "csp__nfilter":[2,3,4,5,6,7,8],
    "svc__C": [0.5,1,1.5],
    "svc__kernel":["rbf","linear"],
}

# DLCSPauto+shLDA
pipelines["DLCSPauto+shLDA"] = make_pipeline(
    Covariances("oas"), CSP(nfilter=6), LDA(solver="lsqr", shrinkage="auto")
)

# CSP+LDA
pipelines["CSP+LDA"] = make_pipeline(
    Covariances("oas"), CSP(nfilter=6), LDA(solver="svd")
)

# MDM
pipelines["MDM"] = make_pipeline(
    Covariances("oas"), MDM(metric="riemann")
)

# TRCSP+LDA
pipelines["TRCSP+LDA"] = make_pipeline(
    Covariances("scm"), TRCSP(nfilter=6), LDA()
)

# LogVariance+LDA
pipelines["LogVariance+LDA"] = make_pipeline(
    LogVariance(),  LDA(solver="svd")
)

# LogVariance+SVM
# pipelines["LogVariance+SVM"] = make_pipeline(
#     LogVariance(), SVC(kernel="linear")
# )
pipelines["LogVariance+SVM"] = Pipeline(
    steps=[
        ("LogVariance", LogVariance()),  
        ("svc", SVC(kernel="linear")),  
    ]
)
param_grid["LogVariance+SVM"] = {
    "svc__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
}

# pipelines={}
# model_list=["Keras_EEGITNet","Keras_EEGNeX","Keras_EEGNet_8_2","Keras_EEGTCNet"]
# for n in model_list:
#     name="./pipelines/"+n+".yml"
#     pipelines[n]=parse_pipelines_from_directory(name)[0]['pipeline']

#----------------------------pipeline definition----------------------------

# experiment setup
from moabb.datasets import BNCI2014_001, Zhou2016,BNCI2014_004

paradigm=MotorImagery(n_classes=2, events=["left_hand", "right_hand"])
ds=[BNCI2014_001(),Zhou2016(),BNCI2014_004()]
seeds=[1,2,3,4,5]
results={}
output_folder="basics/entire_dataset/reruns"
scheme="EntireDatasetEvaluation"
print(f"scheme eval used: {scheme}")

for s in seeds:
    result=run_pipeline(datasets=[BNCI2014_001(),Zhou2016(),BNCI2014_004()], paradigm=paradigm,
                    model_pipeline=pipelines, eval_scheme=scheme,
                    random_state=s, param_grid=param_grid, return_epochs=False)
    print(result)
    for key, df in result.items():
        filename = f"./output_csv/{output_folder}/ds={key}_seed={s}.csv" 
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")
    results[s]=result

    processed_df=process_results(result)
    processed_df.to_csv(f"./output_csv/{output_folder}/summary_seed={s}.csv", index=False)
    print(processed_df)
    
# avg_results=avg_over_seed(results)
# print(avg_results)
# for key, df in avg_results.items():
#     filename = f"./output_csv/{output_folder}/avg_results_ds={key}_seed={seeds}.csv" 
#     df.to_csv(filename, index=False)
#     print(f"Saved {filename}")

