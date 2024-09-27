from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
import warnings
import mne
import moabb
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
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

from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGNetv4
from keras.losses import BinaryCrossentropy
from moabb.pipelines.deep_learning import KerasEEGNeX, KerasEEGTCNet, KerasEEGNet_8_2, KerasEEGITNet, KerasShallowConvNet

moabb.set_log_level("info")
mne.set_log_level("CRITICAL")
warnings.filterwarnings("ignore")

# SCRATCH = os.environ["SCRATCH"]
# SLURM_TMPDIR = os.environ["SLURM_TMPDIR"]
# path = SLURM_TMPDIR+"/mne_data"
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
#----------------------------pipeline definition----------------------------

# experiment setup
paradigm=MotorImagery(n_classes=2, events=["left_hand", "right_hand"])
# extract config from text file for session/subject id
subject_sessions_config=read_config("./benchmark1_subjects.txt")
subject_sessions_setup = extract_subject_sessions(subject_sessions_config)
print(f"Parsed Config for this experiment: {subject_sessions_setup}")
from itertools import islice
filtered_setup=filter_data(paradigm,dict(islice(subject_sessions_setup.items(), 3)))

filtered_ds_cls=[]
for name, setup in filtered_setup.items():
  filtered_ds_cls.append(create_custom_ds(name,setup))

# from moabb.datasets import BNCI2014_001, Zhou2016
#[BNCI2014_001(),Zhou2016()]
result=run_pipeline(datasets=filtered_ds_cls, paradigm=paradigm,
                model_pipeline=pipelines, eval_scheme="WithinSessionEvaluation")
print(result)
processed_df=process_results(result)
print(processed_df)

processed_df.to_csv("./summary_3.csv", index=False)
for key, df in result.items():
    filename = f"./{key}_3.csv" 
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

