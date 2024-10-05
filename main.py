from mne.decoding import CSP
import warnings
import mne
import moabb
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb import benchmark
import os
from utils import *
from dataset_setup import *
from pipeline import *

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.pipeline import Pipeline

from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGNetv4
from keras.losses import BinaryCrossentropy
from moabb.pipelines.deep_learning import KerasEEGNeX, KerasEEGTCNet, KerasEEGNet_8_2, KerasEEGITNet, KerasShallowConvNet

moabb.set_log_level("info")
mne.set_log_level("CRITICAL")
warnings.filterwarnings("ignore")

# SCRATCH = os.environ["SCRATCH"]
# SLURM_TMPDIR = os.environ["SLURM_TMPDIR"]
# from mne import get_config
# from moabb.utils import set_download_dir

# original_path = get_config("MNE_DATA")
# print(f"The download directory is currently {original_path}")
# new_path = SCRATCH+"/mne_data"
# set_download_dir(new_path)
# check_path = get_config("MNE_DATA")
# print(f"Now the download directory has been changed to {check_path}")
path = "./mne_data"
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory '{path}' created for mne data storage.")
else:
    print(f"Directory '{path}' already exists for mne data storage.")

os.environ["MNE_DATA"] = path
os.environ["MOABB_RESULTS"] = path
print(f"Path for MNE_DATA: {os.environ['MNE_DATA']}")

#----------------------------pipeline definition----------------------------
pipelines = {}
# ACM+TS+SVM
pipelines["ACM+TS+SVM"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear")
)

# TS+LR
pipelines["TS+LR"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression()
)

# FgMDM
pipelines["FgMDM"] = make_pipeline(
    Covariances("oas"), MDM(metric="riemann")
)

# TS+SVM
pipelines["TS+SVM"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="rbf")
)

# # TS+EL
# pipelines["TS+EL"] = make_pipeline(
#     Covariances("oas"), TangentSpace(metric="riemann"), ElasticNet()
# )

# FilterBank+SVM
pipelines["FilterBank+SVM"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear")
)

# CSP+SVM
pipelines["CSP+SVM"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear")
)

# DLCSPauto+shLDA
pipelines["DLCSPauto+shLDA"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), LDA()
)

# CSP+LDA
pipelines["CSP+LDA"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), LDA()
)

# MDM
pipelines["MDM"] = make_pipeline(
    Covariances("oas"), MDM(metric="riemann")
)

# TRCSP+LDA
pipelines["TRCSP+LDA"] = make_pipeline(
    Covariances("oas"), TangentSpace(metric="riemann"), LDA()
)

#converts matrix 3D > 2D feature vector.
from moabb.pipelines.features import LogVariance

# LogVariance+LDA
pipelines["LogVariance+LDA"] = make_pipeline(
    Covariances("oas"), LogVariance(),  LDA()
)

# LogVariance+SVM
pipelines["LogVariance+SVM"] = make_pipeline(
    Covariances("oas"), LogVariance(), SVC(kernel="linear")
)

#----------------------------pipeline definition----------------------------

# experiment setup
paradigm=MotorImagery(n_classes=2, events=["left_hand", "right_hand"])
# extract config from text file for session/subject id
subject_sessions_config=read_config("./benchmark1_subjects.txt")
subject_sessions_setup = extract_subject_sessions(subject_sessions_config)
from itertools import islice
filtered_setup=parse_setup(paradigm,subject_sessions_setup)
print(f"Parsed Config for this experiment: {filtered_setup}")

# setup dataset list
filtered_ds=subjDS_setup(paradigm,filtered_setup)
seeds=[1,2,3]
results={}

pipelines = {}

pipelines["TS+SVM"] = Pipeline(
    steps=[("Covariances", Covariances("oas")),
        ("Tangent_Space", TangentSpace(metric="riemann")),
        (
            "svc",
            SVC(kernel="linear"),
        ),
    ]
)
param_grid = {}
param_grid["TS+SVM"] = {
    "svc__C": [0.5,1,1.5],
    "svc__kernel":["rbf","linear"],
}
for s in seeds:
    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=filtered_ds,
        overwrite=True,
        random_state=s,
        hdf5_path="./output_csv/",
        n_jobs=-1,
        save_model=True,
    )
    result = evaluation.process(pipelines, param_grid)
    # result = benchmark(
    #     pipelines="./TSSVM_grid.yml",
    #     evaluations=["WithinSession"],
    #     paradigms=["LeftRightImagery"],
    #     include_datasets=filtered_ds,
    #     results="./output_csv/",
    #     overwrite=True,
    #     plot=False,
    #     output="./output_csv/",
    # )
    print(result)
    filename = f"./output_csv/seed_results/grid_seed={s}.csv" 
    result.to_csv(filename, index=False)
    print(f"Saved {filename}")
# for s in seeds:
#     result=run_pipeline(datasets=filtered_ds, paradigm=paradigm,
#                     model_pipeline=pipelines, eval_scheme="WithinSessionEvaluation",
#                     random_state=s)
#     for ds, output in result.items():
#         df_session_filtered=get_included_df(filtered_setup[ds], output)
#         result[ds]=df_session_filtered
   
#     print(result)
#     save_results_dict(result, seed=s)
#     results[s]=result

#     processed_df=process_results(result)
#     processed_df.to_csv(f"./output_csv/seed_results/summary_seed={s}.csv", index=False)
#     print(processed_df)
    
# avg_results=avg_over_seed(results)
# print(avg_results)
# for key, df in avg_results.items():
#     filename = f"./output_csv/seed_results/avg_results_ds={key}_seed={seeds}.csv" 
#     df.to_csv(filename, index=False)
#     print(f"Saved {filename}")

