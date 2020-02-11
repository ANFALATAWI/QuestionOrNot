#QuestionOrNot
Text classification is the process that involoves taking a piece of text and categorizing it into predefined categories. In this project, a text classification model is implemented using Support Vector Machine (SVM) in order classify a sentence into either a question or not a question. 

## Methodology 
The methodology of this project involved five steps, collecting the data, pre-processing it, extracting its features, then fed it to SVM model. Finally, reporting the evaluation of this model. 

![Methodology of developing question classifier](https://github.com/ANFALATAWI/QuestionOrNot.git)

### 1. Data
the dataset in this project is constructed from the following resources: 
- **Quora:** a question-and-answer  website  where  questions  are  asked,  answered,  and  edited by  users. 8351 question was obtained from this website. For the source of this dataset, go to this [GitHub page](https://github.com/vishaljain3991/cnn_topic_classification/tree/master/data).
- **OPUS:** It is an open source database of sentences and their translations. The collected dataset in this project contained over 10 million English sentences initially, from the Wikipedia website. To find the this dataset, go to [OPUS website](http://opus.nlpl.eu/). 

### 2. Data Preprocessing
The data-set went through several steps in order to clean it and to transform it into an understandable format, to finally obtain a balanced dataset. However, since Quora data-set was already clean, some of these steps were done only on the OPUS data-set
- Removal of short and long sentences.
- Removal of questions.
- Conversion to lower case.
- Removal of punctuation.
- Tokenization.

### 3. Feature Extraction
In order to extract features from text, The Global Vectors for Word Representation (GloVe) was used. The model chosen was the ”GloVe 300-Dimensional Word Vectors Trained on Wikipedia and Gigaword 5 Data” (Pennington, Socher, and Manning 2014). Read [this paper](https://nlp.stanford.edu/pubs/glove.pdf) for more information on glove vectors.

Note: since sentences have different length, padding each one to the maximum sentence length is an important step after extracting their features.

### 4. Model
The classification model that was used in this project is Support Vector Machin (SVM). The hyper parameters chosen for the model were the default parameters. Read [this blog](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72) for more information on the SVM classifier. 

### 5. Result

| Measure |  |
|--|--|
| Training Accuracy | 0.897 |
|Testing Accuracy|0.804|
|Precision|0.87|
|Recall|0.76|
|F1|0.82|

## Environment
Python 3.6+ Jupyter Notebook 6.0.1+

Jupyter Notebook was run on Google Colaboratory (Colab)

### Run
To open the project in Colab:
- open the Notebook file in GitHub and copy the URL (the URL must end in .ipynb): `https://github.com/ANFALATAWI/QuestionOrNot/blob/master/SVM_QuestionOrNot.ipynb`
-  Open [Colab](https://colab.research.google.com).
- From File choose Open notebook (shortcut Ctrl+O), then select the GitHub tab
- Press enter, and select the project from Repository and notebooks

To upload a CSV file from GitHub repository:
- Click on the dataset in a repository, then click on **View Raw**.
- Copy the URL of the raw dataset and store it as a string variable
- load the URL into Pandas read_csv to get the dataframe.

Example:
```python
import pandas as pd
url = 'Raw dataset URL'
df = pd.read_csv(url)
```

Finally, to run the project choose Runtime > Run all (shortcut ctrl+F9)