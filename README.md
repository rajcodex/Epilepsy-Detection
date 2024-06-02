# A Hybrid CNN GRU Model Framework for Epilepsy Detection
Based on our paper "**A Hybrid CNN GRU Model Framework for Epilepsy Detection from EEG Datasets**" will be published in Brain Informatics, Springer.
## Requirements
To install the required dependencies run the following in command prompt: `pip install -r requirements.txt`
## Directory Structure
``` bash
|   main.py
|   proposed_model.py
|   README.md
|   requirement.txt
|
+---dataset
|       mendeley.csv
|       uci.csv
|
+---original datasets
|   +---Mendeley Dataset
|   |   |   New Text Document.txt
|   |   |
|   |   +---setb
|   |   |       O01.txt
|   |   |       O02.txt
|   |   |       O03.txt
|   |   |       O04.txt
|   |   |          .
|   |   |          .
|   |   |       O098.txt
|   |   |       O099.txt
|   |   |
|   |   +---setd
|   |   |       F001.txt
|   |   |       F002.txt
|   |   |       F003.txt
|   |   |       F004.txt
|   |              .
|   |   |          .
|   |   |          .
|   |   |          .
|   |   |          .
|   |   |       F099.txt
|   |   |       F100.txt
|   |   |
|   |   \---sete
|   |           S001.txt
|   |           S002.txt
|   |           S003.txt
|   |           S004.txt
|   |              .
|   |              .
|   |              .
|   |              .
|   |              .
|   |           S099.txt
|   |           S100.txt
|   |
|   \---UCI Dataset
|       +---F
|       |       F001.txt
|       |       F002.txt
|       |       F003.txt
|       |       F004.txt
|       |       N004.TXT
|       |           .
|       |           .
|       |           .
|       |           .
|       |           .
|       |       F099.txt
|       |       F100.txt
|       |
|       +---N
|       |       N001.TXT
|       |       N002.TXT
|       |       N003.TXT
|       |       N004.TXT
|       |           .
|       |           .
|       |           .
|       |           .
|       |           .
|       |       N098.TXT
|       |       N099.TXT
|       |       N100.TXT
|       |
|       +---O
|       |       O001.txt
|       |       O002.txt
|       |       O003.txt
|       |           .
|       |           .
|       |           .
|       |           .
|       |           .

|       |       O099.txt
|       |       O100.txt
|       |
|       +---S
|       |       S001.txt
|       |       S002.txt
|       |       S003.txt
|       |       S004.txt
|       |       S005.txt
|       |       S006.txt
|       |           .
|       |           .
|       |           .
|       |           .
|       |           .
|       |       S099.txt
|       |       S100.txt
|       \---Z
|               Z001.txt
|               Z002.txt
|               Z003.txt
|                  .
|                  .
|                  .
|               Z099.txt
|               Z100.txt
|
\---__pycache__
        proposed_model.cpython-38.pyc
```
The `dataset` directory contains preprocessed csv files for EEG datasets. The `original datasets` contains raw EEG files for all the datasets used in our study. Here, `main.py` consists the code for training and evaluating the model. Lastly, `proposed_model.py` consists the implementation of our proposed framework.
## Running our code
First clone our github repo using this command : `git clone https://github.com/rajcodex/Epilepsy-Detection.git`.Then mount your command prompt or terminal to the directory where `main.py` is present. Lastly, run the following command in your command prompt or terminal.

`python main.py -d <dataset_name> -p <no_classes>`
### Arguments Details
`--d`: It denotes which dataset will be used for training. For UCI dataset put `uci.csv` and for Mendeley Dataset put `mendeley.csv` as argument.
`--p`: It denotes the no. of classes will be taken for classification tasks. For, 2-class classification put 2 and for 5-class classification put 5 as argument.

**Example**: For 2-class classification on UCI dataset, we write the command as follows:

`python main.py -d uci.csv -p 2`

