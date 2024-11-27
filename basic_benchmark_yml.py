import warnings
import mne
import moabb
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from evaluations import benchmark
import os
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

moabb.set_log_level("info")
mne.set_log_level("CRITICAL")
warnings.filterwarnings("ignore")

SCRATCH = os.environ["SCRATCH"]
SLURM_TMPDIR = os.environ["SLURM_TMPDIR"]
from mne import get_config
from moabb.utils import set_download_dir

original_path = get_config("MNE_DATA")
print(f"The download directory is currently {original_path}")
new_path = "./mne_data" #SCRATCH+"/mne_data"
set_download_dir(new_path)
check_path = get_config("MNE_DATA")
print(f"Now the download directory has been changed to {check_path}")

os.environ["MNE_DATA"] = check_path
os.environ["MOABB_RESULTS"] = check_path
print(f"Path for MNE_DATA: {os.environ['MNE_DATA']}")

#for deep convnet:
import tensorflow as tf

tf.config.optimizer.set_jit(False) 

os.environ['TF_DETERMINISTIC_OPS'] = '0'

# experiment setup
from moabb.datasets import BNCI2014_001, Zhou2016,BNCI2014_004
ds=[BNCI2014_001(),Zhou2016(),BNCI2014_004()]
output_folder="basics/entire_dataset"
# "Keras_ShallowConvNet",
model_list=["Keras_DeepConvNet"]
for n in model_list:
    name="./pipelines/"+n+".yml"
    print(f"---running benchmark on {name}---")
    result = benchmark(
        pipelines=name,
        evaluations=["EntireDataset"],
        paradigms=["LeftRightImagery"],
        include_datasets=ds,
        results="./output_csv/net_results/",
        overwrite=True,
        plot=False
    )
    print(result)
    filename = f"./output_csv/{output_folder}/ymlnets_tuned_{n}.csv" 
    result.to_csv(filename, index=False)
    print(f"Saved {filename}")
