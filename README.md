# Towards a foundation model for EEG data
 
---
Abstract:
  This project collaborates with researchers at Mila to develop
  pre-trained transformer models for decoding electroencephalography
  (EEG) data signals, aiming to establish a robust framework for neural
  population dynamics analysis.
  
---

# Background and Related Work

## The MOABB Benchmark

The MOABB (Mother of All BCI Benchmarks) framework addresses critical
challenges in reproducibility within Brain-Computer Interface (BCI)
research. By benchmarking 30 machine learning pipelines across 36
publicly available EEG datasets, MOABB emphasizes paradigms such as
motor imagery. MOABB's open framework offers standardized benchmarking
through uniform data retrieval, preprocessing, and cross-validation.
These practices allow researchers to evaluate model performance
consistently using data's subject-trial pairs, ensuring clear and
unbiased comparisons of model capabilities.

## POYO

POYO is a state-of-the-art framework for large-scale neural decoding. It
leverages a transformer-based architecture with an innovative spike
tokenization strategy, enabling it to model neural population dynamics
across diverse sessions and individuals. By addressing challenges of
neural variability, POYO demonstrates superior transfer learning and
few-shot learning capabilities. This framework significantly improves
over traditional decoding methods, advancing the scalability and
generalizability of neural data analysis.

# Benchmark on Baseline Classifiers

Previous work on MOABB established a reproducible benchmark for
evaluating EEG classification pipelines using standardized datasets. The
benchmark allows clear comparison of model performance across various
paradigms. Establishing benchmark scores using MOABB for POYO is
essential for validating its performance against widely accepted
standards in EEG-based BCI research, particularly for motor imagery
tasks. Such validation is critical to highlight POYO's strengths and its
contributions to advancing neural decoding methodologies.

While MOABB provides a strong foundation, its evaluation methods does
not fully align with POYO's methodologies. The evaluation mechanisms in
MOABB, designed for traditional pipelines, may lack the specificity
needed to assess foundation models like POYO comprehensively. To address
this gap, this part of the project focuses on developing a complementary
evaluation scheme while re-evaluating the baseline classifiers to
establish a reference for POYO's performance on MOABB datasets. This
approach ensures that POYO's unique contributions are effectively
contextualized, thoroughly assessed, and accurately represented within
the framework of standardized benchmarking.

## Datasets and Data Preparation

The primary datasets used in this project are **BNCI2014-001**,
**BNCI2014-004**, and **Zhou2016**, selected due to their superior data
quality compared to other available options. A flexible range of
functions is provided in the `dataset_setup.py` script to enable
customization for filtering and modifying the data. For example,
specifying files such as `BNCI2014-001_1_0.h5`, `BNCI2014-001_1_1.h5`,
or `Zhou2016_3_2.h5` allows the selection of specific datasets and
sessions---e.g., Subject 1's Sessions 0 and 1 from **BNCI2014-001**, and
Subject 3's Session 2 from **Zhou2016**. The setup accounts for
variations in naming conventions for session identifiers, providing
robust functionality for precise data selection.

It is important to note that while filtering subject and session IDs can
also be achieved through MOABB's compound dataset object, this approach
resamples the data to a frequency of 250 Hz by default, which has been
shown to degrade the performance of baseline classifiers. Consequently,
manual filtering is recommended, particularly for identifying and
excluding data from low-quality channels. Additionally, all data
retrieved from MOABB automatically apply a 50 Hz notch filter to
mitigate power-line interference, ensuring baseline signal quality.

## Evaluation Schemes

MOABB provides three evaluation schemes: **Within-Session**,
**Cross-Session**, and **Cross-Subject**.

#### Within-Session Evaluation

This scheme trains and tests a model on data from the same session,
splitting the data into training and testing subsets. Trials are
shuffled before k-fold cross-validation to assess generalization
performance while minimizing overfitting within a single session.

#### Cross-Session and Cross-Subject Evaluations

These evaluations employ a leave-one-out cross-validation approach,
designating one session or subject, respectively, as the test dataset
while training the model on the remaining data. These methods emphasize
transfer learning by addressing variability across subjects and
sessions.

#### Entire-Dataset Evaluation

While the standard schemes are effective for traditional pipelines, they
differ from POYO's approach in several key aspects. To align the
evaluation process with POYO's novel architecture and training
methodology, this project introduces the **Entire-Dataset Evaluation
Scheme**. This approach leverages data from all subjects and sessions
within a dataset.

##### Entire-Dataset Evaluation Workflow:
-   **Outer 5-Fold Cross-Validation:**

    -   *Training Data:* 80% of the shuffled data from all subjects and
        sessions.

    -   *Test Data:* The remaining 20% of the data.

-   **Stratified Inner 3-Fold Cross-Validation (on the Outer Training
    Data):**

    -   *Inner Training Data:* Two-thirds of the outer training data.

    -   *Inner Validation Data:* One-third of the outer training data.
        This step fine-tunes hyperparameters by evaluating different
        configurations.

-   **Model Training and Evaluation:**

    -   After hyperparameter tuning, the model is trained on the entire
        outer training set.

    -   Testing is conducted on the held-out 20% test set, yielding five
        scores (one per fold) rather than scores tied to individual
        subject-session pairs.

This comprehensive evaluation scheme was integrated into MOABB's
benchmarking function, allowing model pipelines to be specified through
YAML configuration files.

## Experiment setup

To ensure uniformity and consistency in evaluating different models, the
pipeline.py script offers a flexible approach, enabling seamless
switching between cognitive task paradigms, datasets, and evaluation
schemes. This flexibility is controlled by a random seed for
reproducibility. For this project, the experiments focused exclusively
on the Left vs. Right Hand motor imagery paradigm. All pipelines
available in MOABB's benchmark table, such as ACM+TS+SVM, CSP+SVM, and
EEGITNet, were utilized for assessment. These pipelines were either
strictly initialized as per the MOABB repository's specifications or
configured using YAML files provided by the framework. The default
evaluation metric employed was ROC AUC score (the area under the ROC
curve scores), suitable for binary classification tasks.

The evaluation's results are averaged across all cross-validation folds,
and the final benchmark results were reported as mean scores and
standard deviations to reflect the model's performance consistency and
reliability.

# Benchmark Results

The benchmarking revealed that most evaluation results closely align
with MOABB's within-session evaluation scores, demonstrating consistency
with the baseline setup.

Entire Dataset Benchmark Results
| Pipeline              | BNCI2014-001 | BNCI2014-004 | Zhou2016   |
|-----------------------|--------------|--------------|------------|
| CSP+LDA              | 77.61±2.47   | 76.26±1.05   | 91.59±1.21 |
| CSP+SVM              | 81.52±1.97   | 77.84±0.87   | 93.0±1.28  |
| DLCSPauto+shLDA      | 77.61±2.48   | 76.26±1.05   | 91.64±1.22 |
| FgMDM                | 83.08±1.37   | 76.16±1.06   | 93.52±1.56 |
| LogVariance+LDA      | 73.78±1.78   | 75.36±1.02   | 90.05±1.36 |
| LogVariance+SVM      | 73.54±1.77   | 75.43±1.0    | 90.22±1.44 |
| MDM                  | 62.13±4.24   | 75.57±1.13   | 86.8±4.06  |
| TRCSP+LDA            | 75.25±3.0    | 76.28±1.04   | 91.78±1.62 |
| TS+EL                | 84.75±1.29   | 76.24±1.04   | 94.24±1.3  |
| TS+LR                | 84.72±1.31   | 76.25±1.04   | 94.31±1.3  |
| TS+SVM               | 88.71±0.8    | 79.39±0.9    | 94.77±1.01 |
| Keras_EEGITNet       | 88.45±1.3    | 74.88±1.84   | 97.19±0.81 |
| Keras_EEGNeX         | 83.59±1.42   | 71.85±1.36   | 95.56±0.94 |
| Keras_EEGNet_8_2     | 89.26±0.94   | 76.99±1.01   | 96.68±0.63 |
| Keras_EEGTCNet       | 90.0±0.35    | 76.75±0.84   | 96.36±0.78 |
| Keras_ShallowConvNet | 91.97±1.34   | 75.32±0.52   | 96.58±1.00 |


Cross Subject Benchmark Results
| Pipeline              | BNCI2014-001 | BNCI2014-004 | Zhou2016   |
|-----------------------|--------------|--------------|------------|
| CSP+LDA              | 76.16±16.05  | 79.37±14.29  | 92.22±5.64 |
| CSP+SVM              | 75.95±15.92  | 73.72±17.14  | 91.07±5.99 |
| DLCSPauto+shLDA      | 76.17±16.07  | 79.37±14.29  | 92.37±5.78 |
| FgMDM                | 76.14±15.57  | 78.43±14.36  | 91.23±5.93 |
| LogVariance+LDA      | 69.77±15.27  | 78.56±14.24  | 89.03±6.60 |
| LogVariance+SVM      | 69.60±15.24  | 78.58±14.26  | 89.15±6.82 |
| MDM                  | 75.95±12.47  | 78.92±14.51  | 93.62±5.17 |
| TRCSP+LDA            | 73.73±14.23  | 78.67±14.32  | 92.35±5.67 |
| TS+EL                | 77.73±15.72  | 78.57±14.33  | 92.41±5.78 |
| TS+LR                | 77.65±15.70  | 78.58±14.32  | 92.32±5.55 |
| TS+SVM               | 76.35±16.43  | 71.92±16.64  | 91.50±5.56 |
| Keras_EEGITNet       | 85.20±8.73   | 75.72±12.56  | 94.66±2.84 |
| Keras_EEGNeX         | 76.42±6.57   | 65.70±12.80  | 92.41±4.65 |
| Keras_EEGNet_8_2     | 82.45±13.19  | 71.30±15.61  | 94.62±3.55 |
| Keras_EEGTCNet       | 82.82±12.84  | 70.53±15.67  | 94.74±2.61 |

Within Session Benchmark Results
| Pipeline              | BNCI2014-001 | BNCI2014-004 | Zhou2016   |
|-----------------------|--------------|--------------|------------|
| CSP+LDA              | 82.66±16.68  | 80.11±15.28  | 93.4±6.98  |
| CSP+SVM              | 82.6±16.7    | 79.14±15.99  | 93.47±7.18 |
| DLCSPauto+shLDA      | 82.74±16.58  | 79.95±15.32  | 92.88±7.15 |
| FgMDM                | 86.33±13.0   | 79.39±15.48  | 92.64±6.63 |
| LogVariance+LDA      | 77.69±16.11  | 78.86±15.01  | 88.54±10.13|
| LogVariance+SVM      | 76.34±17.17  | 78.48±15.38  | 88.32±8.43 |
| MDM                  | 80.83±16.43  | 77.72±16.1   | 90.43±7.22 |
| TRCSP+LDA            | 79.66±16.71  | 79.6±15.71   | 93.17±7.25 |
| TS+EL                | 86.33±13.93  | 80.0±15.38   | 94.72±5.91 |
| TS+LR                | 86.96±13.39  | 80.18±15.29  | 94.56±5.87 |
| TS+SVM               | 86.47±14.1   | 79.45±15.75  | 93.59±6.27 |
| Keras_EEGITNet       | 76.25±15.46  | 66.80±15.98  | 67.91±14.64|
| Keras_EEGNeX         | 69.98±16.11  | 66.40±17.51  | 63.62±16.59|
| Keras_EEGNet_8_2     | 74.79±21.01  | 69.52±19.21  | 92.35±8.78 |
| Keras_EEGTCNet       | 60.76±17.40  | 62.00±18.54  | 77.05±11.85|


