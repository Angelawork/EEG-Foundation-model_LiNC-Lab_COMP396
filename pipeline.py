from moabb.paradigms import MotorImagery
from moabb.datasets import utils
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os
os.environ["MNE_DATA"] = "./mne_data"
os.environ["MOABB_RESULTS"] = "./mne_data"

def run_pipeline(datasets, subjects, paradigm, model_pipeline, eval_scheme):
    """
    Pipeline Model's name will be renamed according to the rename dict.

    Params:
    datasets: list of MOABB dataset
    subjects: list of int
        subject IDs to be used from each dataset
    paradigm: moabb.paradigms object
    model_pipeline: Pipeline or dict
        a pre-built sklearn pipeline or a custom one as dict
    eval_scheme: moabb.evaluations object's name str

    Returns:
    eval_results: dict
        results for each dataset, idx by dataset code

    usage:
    results=run_pipeline(datasets=[Zhou2016(), BNCI2014_001()], subjects=[1, 3] , paradigm=LeftRightImagery(),
             model_pipeline=make_pipeline(CSP(n_components=8), LDA()),
             eval_scheme="WithinSessionEvaluation")
    """
    rename={"lineardiscriminantanalysis":"LDA"}
    eval_results = {}

    for ds in datasets:
      if not paradigm.is_valid(ds):
        raise ValueError(f"{ds.code} not compatible with paradigm {type(paradigm).__name__}")
          
      if hasattr(ds, "get_data"):
        ds.subject_list = subjects
      else:
        raise ValueError(f"Provided dataset invalid: {ds}")

      eval_scheme = WithinSessionEvaluation(paradigm=paradigm, datasets=[ds], overwrite=False)
      if isinstance(model_pipeline, Pipeline):
        pipe_name = "+".join([rename[step.__class__.__name__.lower()] if step.__class__.__name__.lower() in rename 
                                else step.__class__.__name__ 
                                for _, step in model_pipeline.steps])
        pipelines = {pipe_name: model_pipeline}
      elif isinstance(model_pipeline, dict):
        pipelines = model_pipeline
      else:
        raise ValueError(f"Invalid pipeline: {model_pipeline}")
      
      print(f"Running the pipeline with subjects={subjects} on dataset={ds.code} using paradigm: {paradigm}")
      result = eval_scheme.process(pipelines)
      result["pipeline"] = pipe_name
      eval_results[ds.code] = result

    return eval_results

def process_results(results):
    """
    Params:
    results: dict of dataframes
    returned by the run_pipeline() function, key = dataset name with value = corresponding results

    Returns:
    df: dataframe
        In the same format of MOABB benchmark with scores = mean ± std

    """
    df = pd.concat(results.values(), ignore_index=True)
    df=df.groupby(["pipeline", "dataset"])["score"].agg([np.mean, np.std])
    df["scores"] = df.apply(lambda x: f'{x["mean"]:.2f}±{x["std"]:.2f}', axis=1)
    df = df.drop(columns=["mean", "std"]).unstack()
    df.columns = df.columns.droplevel()
    df=df.reset_index()
    return df

if __name__ == "__main__":
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.pipeline import make_pipeline
    import warnings
    import mne
    import moabb
    from moabb.datasets import BNCI2014_001, Zhou2016
    from moabb.evaluations import WithinSessionEvaluation
    from moabb.paradigms import LeftRightImagery
    moabb.set_log_level("info")
    mne.set_log_level("CRITICAL")
    warnings.filterwarnings("ignore")

    results=run_pipeline(datasets=[Zhou2016(), BNCI2014_001()], subjects=[1, 3] , paradigm=LeftRightImagery(),
                model_pipeline=make_pipeline(CSP(n_components=8), LDA()),
                eval_scheme="WithinSessionEvaluation")
    print(results)