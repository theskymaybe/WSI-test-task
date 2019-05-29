# Word Sense Induction using AdaGram and NLTK
This code was written for a pre-employment test task and refers to a part of RUSSE'18 Word Sense Induction shared task.

The baseline algorithm is very simple and uses a [Python implementation](https://github.com/lopuhin/python-adagram) of AdaGram [[Bartunov et al., 2015](http://arxiv.org/abs/1502.07257)] as a pre-trained vector model, and its performance is stable and quite good in terms of the given task.

The simpliest way I found to enhance this algorithm was to filter the input data and therefore make vector representation of the context clearer using the NLTK stopwords list for Russian language. 

This addition boosted the ARI score of the given algorithm significantly: **+0.02 ARI** on the test set is quite an improvement given the datasat and the knowledge-free track we're following. 

## Preparations
This script was tested in Python 3.6 and requires quite a lot of packages to be installed, so run:
```
pip install -r requirements.txt
```
Note that `adagram_nltk.py` and `baseline_adagram.py` contain more requirements inside.

AdaGram should be installed and downloaded separately:

```pip install git+https://github.com/lopuhin/python-adagram.git```

[Link for downloading AdaGram](https://s3.amazonaws.com/kostia.lopuhin/all.a010.p10.d300.w5.m100.nonorm.slim.joblib)

And then placed into the current directory:

```wget 'https://s3.amazonaws.com/kostia.lopuhin/all.a010.p10.d300.w5.m100.nonorm.slim.joblib'```

## Structure of this repository
`data` — a directory containing train and test datasets and outputs for baseline Adagram and AdaGram + NLTK scripts

`adagram_nltk.py` — a script using Adagram + NLTK for label prediction

`baseline_adagram.py` — an Adagram baseline generation script

`evaluate.py` — an evaluation script using Adjusted Rand Score (ARI) as an evaluation metric

`requirements.txt` — the list of the necessary dependencies

## Usage
The usage for `adagram_nltk.py` is the following:
```
python adagram_nltk.py --input INPUT --output OUTPUT --model MODEL
--input  INPUT Path to the input CSV file
--output  OUTPUT Path to the output CSV file
--model  MODEL Path to the AdaGram model file (all.a010.p10.d300.w5.m100.nonorm.slim.joblib in the working directory by default)
```
The `evaluation.py` script can be run in the following way:
```
python evaluate.py --input INPUT
-- input INPUT Path to the CSV with predictions
```

##  The results 
The results are shown in the table below:

 |  **Dataset** | **Baseline AdaGram**      | **AdaGram + NLTK** | 
 |---| ------------- | ------------- |
 |train  | 0.159930  | **0.162855** |
 |  test| 0.195162  | **0.215689**  |
 
 *Table 1. Baseline Adagram vs Adagram + NLTK*

## Experiments and suggestions for the future work

I experimented with some other approaches to the WSI task, for example, I've tried to implement the average word embeddings clustering approach proposed by [[Kutuzov, 2018](https://arxiv.org/ftp/arxiv/papers/1805/1805.02258.pdf)]. I experimented with different clusterization methods and parameters as well as with different datasets, but despite their high score on train part of the dataset, they failed to beat the baseline on the test part. 

Anyway, I found a way to use the average word embeddings clustering approach to enhance my model by blending the best results. Affinity Propagation clusterization method (`damping=0.5, preference=-1.3`), Spectral Clustering as an optional second clustering stage, and differend skipgram models were used in my experiments. If mentioned in the parameters, weighted sum of the context vectors was used. The code isn't present in this repository (yet), but the three best blending results can be found in the following table:

 |  **Dataset & parameters** | **Train ARI**     | **Test ARI** | 
 |---| ------------- | ------------- |
 |Ruscorpora, weighted  | 0,162855 | **0,242398** |
 |Tayga, 2 stages, weighted |**0,168033** | 0,232466 | 
 |RusCorpora + Wiki, vanilla | 0,167468 | 0,232312  |
 
*Table 2. The results of blending*

I've discovered that with these parameters the average word embeddings method deals better with bigger clusters of words, so that might be used in future research. Also there are papers proving that a product of TF/IDF and chi-square of a term with some coefficients gives better performance when it comes to the weighted sum of context vectors, so that might be used as well.
