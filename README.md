# multimodalCKD
This is the official code for the study titled  
"Enhancing Chronic Kidney Disease Diagnosis with Retinal Images through Multimodal Deep Learning Incorporating Urine Dipstick Tests."  
by YoungMin Bhak, Yu Ho Lee, Joonhyung Kim, Kiwon Lee, Daehwan Lee, Eun Chan Jang, Eunjeong Jang, Christopher Seungkyu Lee, Eun Seok Kang, Hyun Wook Han, Sang Min Nam

# Install libraries
```
conda create -n py36 python=3.6 -y
pip install tensorflow-gpu==2.1.0  
pip install tensorflow-addons==0.9.0  
pip install tqdm  
pip install opencv-python==4.5.5.64  
```

# Run the inference step
```
python main.py --filters 128 --arch WRN-40-2 --mode 'eval' --load_dir '2023_02_20_15_38_56_wd0.00003'  --data_type_for_prediction 'test' --hospital_source_type_for_prediction 'CHA' --random_seed 0
```

# Output examples as the CKD probability of two subjects with binocular retinal images
```
# ./models_ckd/preds/2023_02_20_15_38_56_wd0.00003/test_preds_seed0_CHA.out
1.103907972574234009e-01
1.710328310728073120e-01
8.676848560571670532e-02
8.354567736387252808e-02
```