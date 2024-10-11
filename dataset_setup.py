import pandas as pd
import numpy as np
import re
from moabb.datasets.compound_dataset import CompoundDataset

def extract_subject_sessions(input):
    """
    Parse the specific IDs to be used in the experiments
    which have convension datasetname_subj_session into a dict

    input:
    BNCI2014-001_5_1.h5
    output:
    >>> {'BNCI2014-001': [(5, 1)]}
    """
    output = {}
    for t in input:
        parsed = re.match(r"([A-Za-z0-9\-]+)_(\d+)_(\d+)\.h5", t)
        if parsed:
            ds_name, subject_id, session_id = parsed.groups()
            if ds_name not in output:
                output[ds_name] = {}
            if subject_id not in output[ds_name]:
                output[ds_name][subject_id] = []
            output[ds_name][subject_id].append(session_id)
        else:
            print(f"Invalid subject-session format: {t}")
    
    format_output = {ds_name: [(subject_id, session_ids) for subject_id, session_ids in subject_sessions.items()]
                    for ds_name, subject_sessions in output.items()}
    return format_output

def get_dataset_by_name(paradigm, name):
    for idx, ds in enumerate(paradigm.datasets):
        if ds.code == name:
            return idx, ds
    print(f"{name} in {paradigm} not found!")
    return None, None

def parse_setup(paradigm,setup):
    output={}
    for ds, config in setup.items():
        allowed_subj_session = {int(subject):session for subject, session in config}
        allowed_subj = allowed_subj_session.keys()
        
        _, dataset = get_dataset_by_name(paradigm, ds)
        for subj in allowed_subj:
            data=dataset.get_data([subj])
            allowed_sessions = dict(allowed_subj_session).get(subj, [])
            available_sessions = list(data[subj].keys())
            filtered_sessions=[]

            for id1, id2 in zip(allowed_sessions, available_sessions):
                if id1 in id2:
                    filtered_sessions.append(id2)
            allowed_subj_session[subj]=filtered_sessions
        output[ds]=allowed_subj_session
    return output

def subjDS_setup(paradigm, filter_dict):
    """only subjects are filtered"""
    output=[]
    for ds_name in filter_dict.keys():
        _, dataset = get_dataset_by_name(paradigm, ds_name)
        dataset.subject_list = list(filter_dict[ds_name].keys())
        output.append(dataset)
    return output

def compoundDS_setup(paradigm, filter_dict):
  filtered_session_name=[]
  filtered_data={}
  filtered_subject_tuple=[]

  for ds_name in filter_dict.keys():
    p_idx, dataset = get_dataset_by_name(paradigm, ds_name)

    for subj_id, session in filter_dict[ds_name]:
        subj_id=int(subj_id)
        raw_data = dataset.get_data([subj_id])
        available_sessions = list(raw_data[subj_id].keys())
        print(f"Dataset {dataset} given subjects = {subj_id}\navailable sessions: {available_sessions}")

        # some dataset have different naming convension for sessions' names
        # assume target session_id is contained within its corresponding sessions' name
        for s_name in session:
            for a_name in available_sessions:
                if str(s_name) in a_name:
                    filtered_session_name.append(a_name)
                    print(f"Give session_id: {s_name} found in dataset's available sessions: {a_name}")
        filtered_subject_tuple.append((dataset, subj_id,filtered_session_name, None))
        filtered_session_name=[]

    # get tuple for creation of custom dataset class
    filtered_data[ds_name]=filtered_subject_tuple
    filtered_subject_tuple=[]
  return filtered_data

def create_custom_ds(name, setup):
    class CustomDS(CompoundDataset):
        def __init__(self, setup):
            subject_list = setup

            CompoundDataset.__init__(
                self,
                subjects_list=subject_list,
                code=name,
                interval=[0, 1.0]
            )

    return CustomDS(setup)