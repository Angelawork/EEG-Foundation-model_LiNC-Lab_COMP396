import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def process_results(results):
    """
    Params:
    results: dict of dataframes
    returned by the run_pipeline() function, key = dataset name with value = corresponding results

    Returns:
    df: dataframe
        In the same format of MOABB benchmark with scores = mean ± std

    usage example:
    >>>  returned from run_pipeline:
      {'Zhou2016':       score      time  samples subject session  channels  n_sessions   dataset pipeline
      0  0.887248  0.289396    119.0       1       0        14           3  Zhou2016  CSP+LDA
      1  0.922000  0.296401    100.0       1       1        14           3  Zhou2016  CSP+LDA
      2  0.958000  0.379393    100.0       1       2        14           3  Zhou2016  CSP+LDA
      3  0.984000  0.149583    100.0       3       0        14           3  Zhou2016  CSP+LDA
      4  0.980000  0.163201    100.0       3       1        14           3  Zhou2016  CSP+LDA
      5  0.990000  0.169998    100.0       3       2        14           3  Zhou2016  CSP+LDA, 'BNCI2014-001':       score      time  samples subject session  channels  n_sessions       dataset pipeline
      0  0.932721  0.221730    144.0       1  0train        22           2  BNCI2014-001  CSP+LDA
      1  0.950884  0.221004    144.0       1   1test        22           2  BNCI2014-001  CSP+LDA
      2  0.992381  0.218329    144.0       3  0train        22           2  BNCI2014-001  CSP+LDA
      3  0.990272  0.240432    144.0       3   1test        22           2  BNCI2014-001  CSP+LDA}

    >>>  returned df:
      pipeline BNCI2014-001   Zhou2016
      CSP+LDA    0.97±0.03  0.95±0.04
    """
    df = pd.concat(results.values(), ignore_index=True)
    df=df.groupby(["pipeline", "dataset"])["score"].agg([np.mean, np.std])
    df["scores"] = df.apply(lambda x: f'{x["mean"]:.2f}±{x["std"]:.2f}', axis=1)
    df = df.drop(columns=["mean", "std"]).unstack()
    df.columns = df.columns.droplevel()
    df=df.reset_index()
    return df

def plot(results):
  """
  Plotting the results dataframe over the subjects, provided by moabb

  """
  results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]
  g = sns.catplot(
      kind="bar",
      x="score",
      y="subj",
      col="dataset",
      data=results,
      orient="h",
      palette="viridis",
  )
  plt.show()