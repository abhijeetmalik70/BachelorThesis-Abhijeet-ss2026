** Research Question 1 **
* Evaluation of multiple SLMs *

run the requirements.txt file 

-get the api key for the ollama cloud to run the models on cloud : "OLLAMA KEY"

-get the api key of hugging face to download the small size models locally  : "HUGGUNG FACE API KEY"

run the script : evaluation.py 

give the argument file : V1_masters_with_labels.xlsx

This file has the questions , their answers and the ground truth ( both the labels : accuracy and Context & relevance )

Matrix for the evaluation of the multiple models are - MIou , F1 , Precison , Recall and Accuracy 

for evaluting the performance of the model you can choose the models from : models-choice.txt file 

* Fine-tunning of the best SLM for moderation 

pass 






** Research Question 2 **

run the file generation.py




** Research Question 3 **

The file with the generations of answers by the model is given . it can be counted manually as the dataset is small of 100 
total questions 
