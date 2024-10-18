import logging
from copy import deepcopy
from time import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from mne.epochs import BaseEpochs
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation
from moabb.evaluations.utils import create_save_path, save_model_cv, save_model_list


try:
    from codecarbon import EmissionsTracker

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False

log = logging.getLogger(__name__)

# Numpy ArrayLike is only available starting from Numpy 1.20 and Python 3.8
Vector = Union[list, tuple, np.ndarray]

from sklearn.model_selection import train_test_split
class EntireDatasetEvaluation(BaseEvaluation):
    def __init__(self, filter_setup, *args, **kwargs):
        self.filter_setup = filter_setup
        super().__init__(*args, **kwargs)

    def evaluate(self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None):
        run_pipes = {}
        
        # Check which pipelines have not been computed yet
        run_pipes.update(
            self.results.not_yet_computed(pipelines, dataset, "all_subjects", process_pipeline)
        )
        if len(run_pipes) == 0:
            return

        # Get the data from all subjects combined
        X, y, metadata = self.paradigm.get_data(
            dataset=dataset,
            subjects=dataset.subject_list,
            return_epochs=self.return_epochs,
            return_raws=self.return_raws,
            cache_config=self.cache_config,
            postprocess_pipeline=postprocess_pipeline,
        )

        # Apply filter if necessary
        if self.filter_setup is not None:
            ds_name = type(dataset).__name__.replace("_", "-")
            filter_dict = self.filter_setup.get(ds_name, None)
            if filter_dict is not None:
                X_filtered, y_filtered, metadata_filtered = [], [], pd.DataFrame()
                
                for subj in dataset.subject_list:
                    X_subj, y_subj, metadata_subj = self.paradigm.get_data(
                        dataset=dataset,
                        subjects=[subj],
                        return_epochs=self.return_epochs,
                        return_raws=self.return_raws,
                        cache_config=self.cache_config,
                        postprocess_pipeline=postprocess_pipeline,
                    )
                    session_keep = filter_dict.get(subj, [])
                    session_mask = metadata_subj['session'].isin(session_keep)

                    X_filtered.append(X_subj[session_mask])
                    y_filtered.append(y_subj[session_mask])
                    metadata_filtered = pd.concat((metadata_filtered, metadata_subj[session_mask]), ignore_index=True)

                X = np.concatenate(X_filtered) if X_filtered else np.array([])
                y = np.concatenate(y_filtered) if y_filtered else np.array([])
                metadata = metadata_filtered

        # Encode labels
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        # (80-20 split)
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=self.random_state
        # )

        #inner cross-validation for hyperparameter tuning
        inner_cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)
        scorer = get_scorer(self.paradigm.scoring)

        for name, clf in run_pipes.items():
            t_start = time()

            clf = self._grid_search(
                param_grid=param_grid, name=name, grid_clf=clf, inner_cv=inner_cv
            )

            # Train the final model on the training set (X_train)
            model = deepcopy(clf).fit(X_train,y_train)
            print("--------------model fitted-----------------")
            duration = time() - t_start

            # Save the model if necessary
            if self.hdf5_path is not None and self.save_model:
                model_save_path = create_save_path(
                    hdf5_path=self.hdf5_path,
                    code=dataset.code,
                    subject="all_subjects",
                    session="",
                    name=name,
                    grid=False,
                    eval_type="CrossSubject",
                )
                save_model_cv(model=model, save_path=model_save_path)

            test_score = _score(
                estimator=model,
                X_test=X_test,
                y_test=y_test,
                scorer=scorer,
                score_params={},
            )

            # Collect the results
            nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
            res = {
                "time": duration,
                "dataset": dataset,
                "subject": "all_subjects",
                "session": "all_sessions",
                "score": test_score,
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "n_samples": len(X_train)+len(X_test),
                "n_channels": nchan,
                "pipeline": name,
            }
            print(res)

            yield res

    def is_valid(self, dataset):
        return True

class CrossSubjectEvaluation(BaseEvaluation):
    def __init__(
        self,
        filter_setup,
        *args, **kwargs
    ):
        self.filter_setup = filter_setup
        super().__init__(*args, **kwargs)


    # flake8: noqa: C901
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None,

    ):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")

        ds_name = type(dataset).__name__.replace("_","-")
        filter_dict = self.filter_setup.get(ds_name, None)
        # print(f"--------ds_name:{ds_name}------------")
        # print(f"filter_dict:{filter_dict}")

        run_pipes = {}
        for subject in dataset.subject_list:
            run_pipes.update(
                self.results.not_yet_computed(
                    pipelines, dataset, subject, process_pipeline
                )
            )
        if len(run_pipes) == 0:
            return

        # get the data
        print(f"ds subj list:{dataset.subject_list}")
        X, y, metadata = self.paradigm.get_data(
            dataset=dataset,
            subjects=dataset.subject_list,
            return_epochs=self.return_epochs,
            return_raws=self.return_raws,
            cache_config=self.cache_config,
            postprocess_pipeline=postprocess_pipeline,
        )
        print(f"origin X shape: {X.shape}")
        print(f"origin y shape: {y.shape}")
        print(f"origin metadata shape: {metadata.shape}")
        print(f"origin metadata:{metadata}")

        X_filtered, y_filtered, metadata_filtered = [], [], pd.DataFrame()

        for subj in dataset.subject_list:
          X_subj, y_subj, metadata_subj = self.paradigm.get_data(
              dataset=dataset,
              subjects=[subj],
              return_epochs=self.return_epochs,
              return_raws=self.return_raws,
              cache_config=self.cache_config,
              postprocess_pipeline=postprocess_pipeline,
          )
          print(f"--------------before filter:{metadata_subj['session'].unique()}--------------")
          session_keep = filter_dict[subj]
          session_mask = metadata_subj['session'].isin(session_keep)

          X_subj_filtered = X_subj[session_mask]
          y_subj_filtered = y_subj[session_mask]
          metadata_subj_filtered = metadata_subj[session_mask]
          X_filtered.append(X_subj_filtered)
          y_filtered.append(y_subj_filtered)
          metadata_filtered = pd.concat((metadata_filtered, metadata_subj_filtered), ignore_index=True)
          print(f"--------------after filter:{metadata_filtered['session'].unique()}--------------")

        X = np.concatenate(X_filtered) if X_filtered else np.array([])
        y = np.concatenate(y_filtered) if y_filtered else np.array([])
        metadata = metadata_filtered

        print(f"Filtered X shape: {X.shape}")
        print(f"Filtered y shape: {y.shape}")
        print(f"Filtered metadata shape: {metadata.shape}")
        print(f"metadata:{metadata}")
        print("-------------------------------------")

        # encode labels
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        # extract metadata
        groups = metadata.subject.values
        sessions = metadata.session.values
        n_subjects = len(dataset.subject_list)

        scorer = get_scorer(self.paradigm.scoring)

        # perform leave one subject out CV
        if self.n_splits is None:
            cv = LeaveOneGroupOut()
        else:
            cv = GroupKFold(n_splits=self.n_splits)
            n_subjects = self.n_splits

        inner_cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)

        # Implement Grid Search

        if _carbonfootprint:
            # Initialise CodeCarbon
            tracker = EmissionsTracker(save_to_file=False, log_level="error")

        # Progressbar at subject level
        for cv_ind, (train, test) in enumerate(
            tqdm(
                cv.split(X, y, groups),
                total=n_subjects,
                desc=f"{dataset.code}-CrossSubject",
            )
        ):
            subject = groups[test[0]]
            # now we can check if this subject has results
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            # iterate over pipelines
            for name, clf in run_pipes.items():
                if _carbonfootprint:
                    tracker.start()
                t_start = time()
                clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=clf, inner_cv=inner_cv
                )
                model = deepcopy(clf).fit(X[train], y[train])
                if _carbonfootprint:
                    emissions = tracker.stop()
                    if emissions is None:
                        emissions = 0
                duration = time() - t_start

                if self.hdf5_path is not None and self.save_model:
                    model_save_path = create_save_path(
                        hdf5_path=self.hdf5_path,
                        code=dataset.code,
                        subject=subject,
                        session="",
                        name=name,
                        grid=False,
                        eval_type="CrossSubject",
                    )

                    save_model_cv(
                        model=model, save_path=model_save_path, cv_index=str(cv_ind)
                    )
                # we eval on each session
                for session in np.unique(sessions[test]):

                    ix = sessions[test] == session

                    score = _score(
                        estimator=model,
                        X_test=X[test[ix]],
                        y_test=y[test[ix]],
                        scorer=scorer,
                        score_params={},
                    )

                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": session,
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }

                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)
                    yield res

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1




import logging
import os
import os.path as osp
from pathlib import Path

import mne
import pandas as pd
import yaml

from moabb import paradigms as moabb_paradigms
from moabb.analysis import analyze
from moabb.evaluations import (
    CrossSessionEvaluation,
    WithinSessionEvaluation,
)
from moabb.pipelines.utils import (
    generate_paradigms,
    generate_param_grid,
    parse_pipelines_from_directory,
)


try:
    from codecarbon import EmissionsTracker  # noqa

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False

log = logging.getLogger(__name__)


def benchmark(  # noqa: C901
    pipelines="./pipelines/",
    evaluations=None,
    paradigms=None,
    results="./results/",
    overwrite=False,
    output="./benchmark/",
    n_jobs=-1,
    plot=False,
    contexts=None,
    include_datasets=None,
    exclude_datasets=None,
    n_splits=None,
    cache_config=None,
    optuna=False,
):

    eval_type = {
        "WithinSession": WithinSessionEvaluation,
        "CrossSession": CrossSessionEvaluation,
        "CrossSubject": CrossSubjectEvaluation,
        "EntireDataset": EntireDatasetEvaluation
    }

    mne.set_log_level(False)
    # logging.basicConfig(level=logging.WARNING)

    output = Path(output)
    if not osp.isdir(output):
        os.makedirs(output)

    pipeline_configs = parse_pipelines_from_directory(pipelines)

    context_params = {}
    if contexts is not None:
        with open(contexts, "r") as cfile:
            context_params = yaml.load(cfile.read(), Loader=yaml.FullLoader)

    prdgms = generate_paradigms(pipeline_configs, context_params, log)
    if paradigms is not None:
        prdgms = {p: prdgms[p] for p in paradigms}

    param_grid = generate_param_grid(pipeline_configs, context_params, log)

    log.debug(f"The paradigms being run are {prdgms.keys()}")

    if len(context_params) == 0:
        for paradigm in prdgms:
            context_params[paradigm] = {}

    # Looping over the evaluations to be done
    df_eval = []
    for evaluation in evaluations:
        eval_results = dict()
        for paradigm in prdgms:
            # get the context
            log.debug(f"{paradigm}: {context_params[paradigm]}")
            p = getattr(moabb_paradigms, paradigm)(**context_params[paradigm])
            # List of dataset class instances
            datasets = p.datasets
            d = _inc_exc_datasets(datasets, include_datasets, exclude_datasets)
            log.debug(
                f"Datasets considered for {paradigm} paradigm {[dt.code for dt in d]}"
            )

            ppl_with_epochs, ppl_with_array = {}, {}
            for pn, pv in prdgms[paradigm].items():
                if "braindecode" in pn or "Keras" in pn:
                    ppl_with_epochs[pn] = pv
                else:
                    ppl_with_array[pn] = pv

            if len(ppl_with_epochs) > 0:
                # Braindecode pipelines require return_epochs=True
                context = eval_type[evaluation](
                    paradigm=p,
                    datasets=d,
                    random_state=42,
                    hdf5_path=results,
                    n_jobs=n_jobs,
                    overwrite=overwrite,
                    return_epochs=True,
                    n_splits=n_splits,
                    cache_config=cache_config,
                    filter_setup=None,
                )
                paradigm_results = context.process(
                    pipelines=ppl_with_epochs, param_grid=param_grid
                )
                paradigm_results["paradigm"] = f"{paradigm}"
                paradigm_results["evaluation"] = f"{evaluation}"
                eval_results[f"{paradigm}"] = paradigm_results
                df_eval.append(paradigm_results)

            # Other pipelines, that use numpy arrays
            if len(ppl_with_array) > 0:
                context = eval_type[evaluation](
                    paradigm=p,
                    datasets=d,
                    random_state=42,
                    hdf5_path=results,
                    n_jobs=n_jobs,
                    overwrite=overwrite,
                    n_splits=n_splits,
                    cache_config=cache_config,
                    filter_setup=None,
                )
                paradigm_results = context.process(
                    pipelines=ppl_with_array, param_grid=param_grid
                )
                paradigm_results["paradigm"] = f"{paradigm}"
                paradigm_results["evaluation"] = f"{evaluation}"
                eval_results[f"{paradigm}"] = paradigm_results
                df_eval.append(paradigm_results)

        # Combining FilterBank and direct paradigms
        eval_results = _combine_paradigms(eval_results)

        _save_results(eval_results, output, plot)

    df_eval = pd.concat(df_eval)
    _display_results(df_eval)

    return df_eval


def _display_results(results):
    """Print results after computation."""
    tab = []
    for d in results["dataset"].unique():
        for p in results["pipeline"].unique():
            for e in results["evaluation"].unique():
                r = {
                    "dataset": d,
                    "evaluation": e,
                    "pipeline": p,
                    "avg score": results[
                        (results["dataset"] == d)
                        & (results["pipeline"] == p)
                        & (results["evaluation"] == e)
                    ]["score"].mean(),
                }
                if _carbonfootprint:
                    r["carbon emission"] = results[
                        (results["dataset"] == d)
                        & (results["pipeline"] == p)
                        & (results["evaluation"] == e)
                    ]["carbon_emission"].sum()
                tab.append(r)
    tab = pd.DataFrame(tab)
    print(tab)


def _combine_paradigms(prdgm_results):
    """Combining FilterBank and direct paradigms.

    Applied only on SSVEP for now.

    Parameters
    ----------
    prdgm_results: dict of DataFrame
        Results of benchmark for all considered paradigms

    Returns
    -------
    eval_results: dict of DataFrame
        Results with filterbank and direct paradigms combined
    """
    eval_results = prdgm_results.copy()
    combine_paradigms = ["SSVEP"]
    for p in combine_paradigms:
        if f"FilterBank{p}" in eval_results.keys() and f"{p}" in eval_results.keys():
            eval_results[f"{p}"] = pd.concat(
                [eval_results[f"{p}"], eval_results[f"FilterBank{p}"]]
            )
            del eval_results[f"FilterBank{p}"]
    return eval_results


def _save_results(eval_results, output, plot):
    """Save results in specified folder.

    Parameters
    ----------
    eval_results: dict of DataFrame
        Results of benchmark for all considered paradigms
    output: str or Path
        Folder to store the analysis results
    plot: bool
        Plot results after computing
    """
    for prdgm, prdgm_result in eval_results.items():
        prdgm_path = Path(output) / prdgm
        if not osp.isdir(prdgm_path):
            prdgm_path.mkdir()
        analyze(prdgm_result, str(prdgm_path), plot=plot)


def _inc_exc_datasets(datasets, include_datasets, exclude_datasets):
    d = list()
    if include_datasets is not None:
        # Assert if the inputs are key_codes
        if isinstance(include_datasets[0], str):
            # Map from key_codes to class instances
            datasets_codes = [d.code for d in datasets]
            # Get the indices of the matching datasets
            for incdat in include_datasets:
                if incdat in datasets_codes:
                    d.append(datasets[datasets_codes.index(incdat)])
        else:
            # The case where the class instances have been given
            # can be passed on directly
            d = list(include_datasets)
        if exclude_datasets is not None:
            raise AttributeError(
                "You could not specify both include and exclude datasets"
            )

    elif exclude_datasets is not None:
        d = list(datasets)
        # Assert if the inputs are not key_codes i.e. expected to be dataset class objects
        if not isinstance(exclude_datasets[0], str):
            # Convert the input to key_codes
            exclude_datasets = [e.code for e in exclude_datasets]

        # Map from key_codes to class instances
        datasets_codes = [d.code for d in datasets]
        for excdat in exclude_datasets:
            del d[datasets_codes.index(excdat)]
    else:
        d = list(datasets)
    return d