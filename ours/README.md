
For generating results on GAIT dataset, cd gait and follow instructions in gait/README.md


## Step 1: Download trained models:
 https://drive.google.com/file/d/1uczjf4jjmfd7hgqqGeqAg6EWu8XydO3S/view?usp=drive_link

## Step 2: Download data (gait dataset-https://drive.google.com/file/d/1KRqNs4rQQpsl6s-TyrdeqaemarjBV7-J/view?usp=drive_link)(https://drive.google.com/drive/folders/12ssAwZ8BhOiXMypEFq6FHsx8Ui4nPT0Z?usp=drive_link)
     unzip data.zip
      
## Step 3: Download the two folders drift_log containing pre-computed fisher and p-values for 1 run for speedy evaluation: [https://drive.google.com/drive/folders/1o2bQ6M17kvN6b78KYPuAv0oavZ0Mf926?usp=sharing](https://drive.google.com/drive/folders/1zgZCKfLdav0c0jBQzuuV31L56F03hCo7?usp=drive_link)
### Note: Rename the downloaded zip files to drift_log.zip respectively
      Drift: unzip drift_log.zip
      
### Install requirements 
      pip install -r requirements.txt
      
# Step 4: Generate the following results after populating --gpu

## Generate AUROC and TNR results for Replay OODs

      python3 check_OOD_carla.py --gpu 0 --cuda --ckpt saved_models/carla_model.pt --n 20 --out_folder_name out_replay/out --save_dir carla_log/replay --printTNR 1 --transformation_list speed shuffle reverse periodic identity

## Generate AUROC and TNR results for Drift OODs 
      python3 check_OOD_drift.py --gpu 0 --cuda --ckpt saved_models/drift.pt --n 20 --save_dir drift_log --transformation_list speed shuffle reverse periodic identity
### Train VAE model for precition of the applied transformation on the drift dataset
     python3 train_drift.py --cl 16 --log saved_models --bs 2 --gpu 0 --lr 0.00001


## This code is built on changes from original implemenattion and correction to make it running 
https://github.com/kaustubhsridhar/time-series-OOD.git
    
