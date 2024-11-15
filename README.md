# CoDIT
CS637 Group Project: Paper implementation and improvement of Conformal Out-of-Distribution Detection in Time Series Data Ramneet Kaur, Kaustubh Sridhar, Sangdon Park, et al.

Group Members: Arpit Raj(210192), Danish Vasdev(210298), Prakhar Mishra(210738), Priyanshu Raj Jindal(210787), Rohan Virmani(210871), Ujjawal Dubey(211121)
## Files and Implementation
### Reproducing Results and approach (e value method instead of p value)
The code available has been corrected and is up and running. It can be run directly from 'code_run.ipynb' file after loading the required datasets from the links provided (the GAIT dataset is small and can be downloaded for use on kaggle).

A new file named '/ours/gait/check_ood_gait_new.py' has been added. It's an alternative approach which uses e-values instead of p-value directly for out-of-distribution detection. The model/training steps were not modified to ensure that the approach mentioned in the paper can be directly extended to other datasets. Our work, however, is primarily concerned with the GAIT dataset.
### Extending current approach to new application on FGSM attack and presentaing an alternative approach which uses activation function for NCM. 
AWS_Utilization.ipynb: For the FGSM experiment, activation function NCM experiment

This extends the application of OOD detection using CoDIT to FGSM attacks.
