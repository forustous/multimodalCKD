# multimodalCKD
This is the official code for the study titled  
"Enhancing Chronic Kidney Disease Diagnosis with Retinal Images through Multimodal Deep Learning Incorporating Urine Dipstick Tests."  
by YoungMin Bhak, Yu Ho Lee, Joonhyung Kim, Kiwon Lee, Daehwan Lee, Eun Chan Jang, Eunjeong Jang, Christopher Seungkyu Lee, Eun Seok Kang, Hyun Wook Han, Sang Min Nam

# Install libraries
```
conda create -n py36 python=3.6 -y
pip install pandas  
pip install tensorflow-gpu==2.1.0  
pip install opencv-python==4.5.5.64  
pip install pillow  
pip install scikit-image  

# Optional
# If you want to use (multiple) GPUs
# Install TensorRT and CUDA 10.1
# pip install tensorflow-addons==0.9.0  
# pip install tqdm  
```

# Run the inference step
```
conda activate py36
python ckd.py
```

# The process from running the code to obtaining the predicted probability.
- When you run the ckd.py script, the raw retinal image (image.jpg) and urine analysis data (urine.csv) are preprocessed.  
- The preprocessed multimodal data is then passed to the trained CKD model, resulting in a predicted CKD probability.

# Output example from the CKD model
```
The probability of CKD is 0.469.
```