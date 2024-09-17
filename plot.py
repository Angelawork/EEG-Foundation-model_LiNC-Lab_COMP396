import matplotlib.pyplot as plt
import seaborn as sns

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