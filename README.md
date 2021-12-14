# MLGWSC-1 - Machine Learning Gravitational-Wave Search (Mock Data) Challenge

## Introduction

Welcome to the first machine learning gravitational-wave search mock 
data challenge hosted by the Albert-Einstein-Institut Hannover and the 
Friedrich-Schiller Universität Jena. In this challenge participants are 
tasked with finding gravitational-wave signals of varying complexity in 
a noisy background. Entries are evaluated on metrics which are used to 
classify the performance of real-world, state-of-the-art search 
algorithms.

The goal of this challenge is to create a collaborative publication that 
collects state-of-the-art machine learning based gravitational-wave 
search algorithms and enables a comparison to classical approaches 
such as matched filtering or coherent burst searches. Through this, we 
strive to highlight the advantages of different entries for specific 
tasks and want to pinpoint areas where further research seems fruitful.

Because this is a collaborative work, all teams that submit an algorithm 
and choose not to retract it before final publication will gain 
co-authorship. We nonetheless encourage publications on the individual 
algorithms to describe details of pre-processing, post-processing, 
training, etc. We, furthermore, encourage the publication of the source 
code used for training and evaluation to foster reproducability. 
However, open source code is not required for submission.

Although this challenge is focused on machine learning approaches, we do 
accept submissions which do not make use of this relatively new area of 
research.

If you want to partipate in this mock data challenge, please get in 
contact with us by sending a mail to [mlgwsc@aei.mpg.de](mailto:mlgwsc@aei.mpg.de).
We accept registrations up to a maximum number of 30 participating 
groups until December 31st, 2021. The deadline for the final submission 
of the algorithm is March 31st, 2022.

On submission, we will evaluate your algorithm on a validation set. The 
performance on this validation set will then be reported back to you to 
check that the algorithm behaves as expected. Once we have confirmation 
by the group that the algorithm performs within the expected margins of 
error, we will evaluate the submission on a secret test set that is the 
same for all entries. The performance on this set will only be reported 
back to the groups on the first circulation of the publication draft. 
Submissions may be retracted at any point prior to final publication of 
the manuscript. For more information please refer to [this page](https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Submission-Details).

## Contents of this Repository

This repository contains source code to generate data of the kind that 
will be used for final evaluation as well as the source code that will 
be used to carry out this final evaluation. It also contains a few 
configuration files that are required for data generation.

Submissions must be able to process a file of HDF5 format that contains 
the raw strain data for 2 detectors. Any required pre-processing is 
expected to be performed by the submitted code. The output is expected 
to be another file of HDF5 format which contains times, 
ranking-statistic like values, and timing accuracies for candidate 
events. The ranking-statistic like values are numbers where a larger 
value is supposed to correspond to a larger probability of an 
astrophysical event to be present. For details on the input- and
output-format please refer to the [Wiki](https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Submission-Details#algorithm-inputoutput-format) of this repository.

## Requirements
To run the code you need to have a working installation of Python 3.7 or
higher. You will then need to install dependencies using
```
pip install -r requirements.txt
```
This installs a version of the PyCBC github that was tested and
confirmed to be working. Older versions may be missing required functions.

For more detailed installation instructions please refer to [this page](https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Provided-Software#requirements).

## Citation
If you make use of the code in this repository please cite it
accordingly. For the citation please use the bibtex-entry as well as DOI
provided for each release at [Zenodo](https://zenodo.org/).

You can use the data provided at the badge below for citation purposes:
[![DOI](https://zenodo.org/badge/387493531.svg)](https://zenodo.org/badge/latestdoi/387493531)
