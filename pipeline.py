import utils
import dataset_setup

from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

def run_pipeline(datasets, paradigm, model_pipeline, eval_scheme, random_state=24):
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
    random_state: int
        For CV result's reproducibility
        If not None, can guarantee same seed for shuffling examples

    Returns:
    eval_results: dict
        results for each dataset, idx by dataset code
    """
    rename={"lineardiscriminantanalysis":"LDA"}
    eval_results = {}

    for ds in datasets:
        if eval_scheme == "WithinSessionEvaluation":
            evaluation = WithinSessionEvaluation(paradigm=paradigm, random_state=random_state,
                                                  datasets=[ds], overwrite=True)
        elif eval_scheme =="CrossSessionEvaluation":
            evaluation = CrossSessionEvaluation(paradigm=paradigm, random_state=random_state,
                                                 datasets=[ds], overwrite=True)
        elif eval_scheme =="CrossSubjectEvaluation":
            evaluation=CrossSubjectEvaluation(paradigm=paradigm, random_state=random_state,
                                               datasets=[ds], overwrite=True)
        else:
            raise ValueError(f"Invalid eval_scheme: {eval_scheme}")

        if isinstance(model_pipeline, Pipeline):
            pipe_name = "+".join([rename[step.__class__.__name__.lower()] if step.__class__.__name__.lower() in rename 
                                    else step.__class__.__name__ 
                                    for _, step in model_pipeline.steps])
            pipelines = {pipe_name: model_pipeline}
        elif isinstance(model_pipeline, dict):
            pipelines = model_pipeline
        else:
            raise ValueError(f"Invalid pipeline: {model_pipeline}")
        
        print(f"Running the pipeline on dataset={ds.code} using paradigm: {paradigm}")
        result = evaluation.process(pipelines)
        print(f"Result on dataset={ds.code}: {result}")

        try:#renaming subject id by original order
            original_subject_ids = [subj[1] for subj in ds.subjects_list]
            if isinstance(model_pipeline, Pipeline):
                result["pipeline"] = pipe_name
                result["subject"] =original_subject_ids[:len(result)]
            else:
                for pipe_name in pipelines:
                    result["subject"] = original_subject_ids * (len(result)//len(original_subject_ids))
        except:
            print(f"Failed to rename Dataset {ds.code} pipeline output with original subject id")
        eval_results[ds.code] = result
        
    return eval_results

""" Usage example
subject_sessions=read_config("./benchmark1_subjects.txt")
subject_sessions_setup = extract_subject_sessions(subject_sessions)

paradigm=MotorImagery(n_classes=2, events=["left_hand", "right_hand"])
test_ds_setup={"BNCI2014-001":subject_sessions_setup["BNCI2014-001"][:3],
      "PhysionetMotorImagery":subject_sessions_setup["PhysionetMotorImagery"][:3]}
filtered_setup=compoundDS_setup(paradigm,test_ds_setup)

filtered_ds_cls=[]
for name, setup in filtered_setup.items():
  filtered_ds_cls.append(create_custom_ds(name,setup))
example_pipeline = make_pipeline(CSP(), LDA())

result=run_pipeline(datasets=filtered_ds_cls, paradigm=paradigm,
                model_pipeline=example_pipeline, eval_scheme="WithinSessionEvaluation")
print(process_results(result))
"""


