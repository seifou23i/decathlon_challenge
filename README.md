# ML Engineer case study : Decathlon

This repository proposes a solution for the Decathlon challenge

The Readme file is organized as follows:

* [1 Challenge statement](#Challenge-statement)
    * [1.1 General instructions](#General-instructions)
    * [1.2 Data](#Data)
    * [1.3 Preliminary questions & EDA](#Preliminary-questions-&-EDA)
    * [1.4 Modeling](#Modeling)
    * [1.5 ML pipeline](#ML-pipeline)
* [2 Installation](#installation)
    * [2.1 Setup a Virtualenv (optional)](#setup-a-virtualenv-optional)
    * [2.2 Install from source](#install-from-source)
* [3 Getting Started](#getting-started)
* [4 Documentation](#documentation)

## Challenge statement

### General instructions

The general instructions are as follows:

- Please do the exercise using Python and your favorite IDE. If you think you have better tools, we will be happy to
  hear.
- Export your work to some place where we can see your progress (your commits), for example Github. Please make sure you
  give viewing permission to the people on this email.
- Please make regular commits with your ongoing work. Do not upload the final result when you get it done. Instead, do
  your work in the public so we can follow your regular updates.

The questions are listed below. Please have a look, and do not hesitate to ask any question about the format of the data
files, or the requirements of the questions.

### Data

train.csv.gz & test.csv.gz files contain weekly store-department turnover data (test.csv.gz does not contain the
turnover feature). For confidentiality reasons the turnover values are rescaled. bu_feat.csv.gz file contains some
useful store information.

### Preliminary questions & EDA

    a. Which department made the highest turnover in 2016?
    b. What are the top 5 week numbers (1 to 53) for department 88 in 2015 in terms of turnover over all stores?
    c. What was the top performer store in 2014?
    d. Based on sales can you guess what kind of sport represents department 73?
    e. Based on sales can you guess what kind of sport represents department 117?
    f. What other insights can you draw from the data? Provide plots and figures if needed. (Optional)

### Modeling

In stores many decisions are made by managers at the department level. In order to help store managers in making
mid-term decisions driven by economic data, we want to forecast the turnover for the next 8 weeks at store-department
level.

    a. Build and evaluate performances of a simple estimator able to predict the turnover of test.csv.gz data. 
    The goal here is not to produce a state-of-the-art model of time series forecast
    b. Propose another model or strategy that may increase the quality of the predictions. (Optional)

### ML pipeline

Based on the previous steps of data exploration and modeling and in order to deploy your model in production, develop a
machine learning pipeline which:

    a. Reads the raw data
    b. Transforms the data in the right format for the model
    c. Trains the model
    d. Make predictions and exposes the results

The aim here is to realize a reproducible pipeline.

    e. Describe some common issues involved in the deployment of machine learning models.
    f. Propose a solution which monitors the the model performance in production (optional)

## Installation

### Requirements

- Python >= 3.9

### Setup a Virtualenv (optional)

#### Create a virtual environment

```commandline
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv my_venv
```

#### Enter virtual environment

```commandline
source my_venv/bin/activate
```

### Install from source

```commandline
git clone https://github.com/seifou23i/decathlon_challenge.git
cd decathlon_challenge
pip3 install -U .
cd ..
```

## Getting Started

The answers to the challenge are provided in the form of notebooks, and are accessible via
[getting_started](getting_started) directory.

- [00_Preliminary_Qestions_&_EDA.ipynb](getting_started/00_Preliminary_Qestions_&_EDA.ipynb) : The first part of the challenge which
  covers sql queries and EDA queries
- [01_Modeling.ipynb](getting_started/01_Modeling.ipynb) : The second part of the challenge which
  covers turnover forecasting
- [02_ML_pipeline.ipynb](getting_started/02_ML_pipeline.ipynb) : The last part of the challenge which
  covers ML pipeline development and orchestration using kubeflow
## Documentation

TODO : add docstrings and generate documentation

To generate locally the documentation:

```commandline
pip install sphinx
pip install sphinx-rtd-theme
cd docs
make clean
make html
```
