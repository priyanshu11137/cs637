# The following instructions for generating results  on GAIT dataset. This code has been run in kaggle notebook attached

## Step 1: Download gait-in-neurodegenerative-disease-database-1.0.0 folder from [https://drive.google.com/drive/folders/1Z-3YnlhcCxI_KlFF6FF7tMp5MSEZURRH?usp=sharing](https://drive.google.com/file/d/1KRqNs4rQQpsl6s-TyrdeqaemarjBV7-J/view?usp=drive_link)

## Step 2: Download the three trained models (gait_16.pt, gait_18.pt, gait_20.pt) from [https://drive.google.com/drive/folders/1p0F2D3oTUgB3QKq_0uLu1F9eKRzRibml?usp=sharing](https://drive.google.com/drive/folders/1fflUH_o-_6rNhmZKlt1CauhUJWGKs5p0?usp=drive_link)



### Resolving dependencies
    pip install -r requirement.txt
    
## Step 4: Generate CODiT results in Table 4 after populating the command-line arguments: --ckpt=saved_models/gait_$wl$.pt where wl = 16/18/20, --wl=16/18/20 (same as wl in saved_models/gait_$wl$.pt), and --disease\_type=als/park/hunt/all
    
## Generate baseline results in Table 4 with --wl=16/18/20, --disese\_type=als/hunt/park/all
    python3 check_OOD_baseline.py --disease_type $disease_type$ --wl $wl$ --root_dir data/gait-in-neurodegenerative-disease-database-1.0.0

## Training VAE model on GAIT dataset on wl=16/18/20
    python3 train_gait.py --log saved_models --transformation_list high_pass low_high high_low identity --wl $wl$

