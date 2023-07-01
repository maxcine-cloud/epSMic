# epSMic
We introduced epSMic (effect predictor of synonymous mutation in cancer), a machine learning approach that predicts the effect of synonymous mutations in cancer. We take the COSMIC data as the benchmark dataset, and use the word embedding sequence information, physicochemical property information, and basic information such as conservation and clipping as basic features. Further, we generate advanced features through multi-feature fusion and iteration as our model. Evaluated on biologically validated and widely reported, observed data, it outperforms existing methods and achieves better performance. The model points out that cancer-driving synonymous mutations differ from the effects of disease-causing synonymous mutations, and that there are certain similarities between the effects of other types of cancer and synonymous mutations.
The details are summarized as follows. 

* data: it contains the four types of data described in the paper. In this paper, we used COSMIC as the benchmark dataset, SomaMutDB and synony_valid as independent test sets, respectively, and explored the effects of synonymous mutations in breast cancer by TCGA.

* out: it contains intermediate output result files, including scoring of test and training data and comparison of dimensionality reduction methods.

* plot: it contains the various chart files referred to in the paper.

* src: it contains the code used in the project, including the process of training and testing the model.

* word2Vec: the encoding of the sequence data used.

* model: models used and saved during the project. It contains data preprocessing, intermediate model and final iterative process model. These are in three different folders:

  * data_processing_model

  * intermediate_model_all

  * iterUltimate_model_all

    
## Environment setup
We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/). See the file for detailed environment Settings at  `/environment/environment.yml` 

* python 3.9.10
* scikit-learn 1.0.2
* xgboost  1.5.1

To create a conda environment for epSMic, you can import the entire environment for:

```
conda env create -f environment.yml
```



## Usage

Please see the template data at `/data` ,it contains various characteristic data and synonymous mutations in the form of VCF. If you are trying to using epSMic with your own data, please process you data into the same format as it. Before using our model, you can read the help documentation.

```
python src/main.py -h

usage: main.py [-h] [-dataType DATATYPE] [-dbName dbName] [-dataPath DATAPATH] [-processingmodelPath PROCESSINGMODELPATH] [-intermodelPath INTERMODELPATH] [-itermodelPath ITERMODELPATH] [-processedPath PROCESSEDPATH] [-interdataPath INTERDATAPATH] [-iterdataPath ITERDATAPATH] [-gridPath GRIDPATH]
```



```
python src/main.py 

    optional arguments:
      -h\, --help            show this help message and exit
      -dataType DATATYPE\    test or train
      -dbName dbName\    synony_valid or SomaMutDB or COSMIC
      -dataPath DATAPATH\    data path
      -processingmodelPath PROCESSINGMODELPATH\
                            processing model path
      -intermodelPath INTERMODELPATH\
                            intermediate model path
      -itermodelPath ITERMODELPATH\
                            iterUltimate model path
      -processedPath PROCESSEDPATH\
                            processed data path
      -interdataPath INTERDATAPATH\
                            intermediate data path
      -iterdataPath ITERDATAPATH\
                            iterUltimate data path

```



## Examples

In epSMic, we have listed four types of data, including TCGA, SomaMutDB, COSMIC, and data with experimentally validated and important influence mechanisms. If you want to use these data to run our model, you can follow the example below to get started faster.

1. If you want to get the score in synony_valid via epSMic, you can ideally run like this:

   ```
   python src/main.py -dbName EOSM
   ```

2. If you want to get the score in SomaMutDB via epSMic, you can ideally run like this:

   ```
   python src/main.py -dbName SomaMutDB
   ```

3. If you want to get the score in COSMIC via epSMic, you can ideally run like this:

   ```
   python src/main.py -dbName COSMIC
   ```

5. If you want to retrain a new model, you can get it by training COSMIC data, of course, it will take a certain amount of time, please be patient.

   ```
   python src/main.py -dataType train -dbName COSMIC
   ```



## Reference

```
TO DO
```

