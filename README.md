Hybrid Kolmogorov-Arnold Networks for Medical Image Segmentation


Medical image segmentation plays a vital role in diagnosis and treatment planning, but remains challenging due to the inherent complexity and variability of medical images, especially in capturing non-linear relationships within the data. We propose U-KABS, a novel hybrid framework that integrates the expressive power of Kolmogorov-Arnold Networks (KANs) with a U-shaped encoder-decoder architecture to enhance segmentation performance. The U-KABS model combines the convolutional and squeeze-and-excitation stage, which enhances channel-wise feature representations, and the KAN Bernstein Spline (KABS) stage, which employs learnable activation functions based on Bernstein polynomials and B-splines. This hybrid design leverages the global smoothness of Bernstein polynomials and the local adaptability of B-splines, enabling the model to effectively capture both broad contextual trends and fine-grained patterns critical for delineating complex structures in medical images. Skip connections between encoder and decoder layers support effective multi-scale feature fusion and preserve spatial details. Evaluated across diverse medical imaging benchmark datasets, U-KABS demonstrates superior performance compared to strong baselines, particularly in segmenting complex anatomical structures.



read the scripts file to run the code.


Step 1: 

git clone https://github.com/bhattacharyyadeep/UKABS-

Step 2: Arrange the datasets 
#UKABS
├── inputs
│   ├── Dataset_name
│     ├── images.
|     ├── masks
│        ├── 0

and install all the requirement from requirements_2.txt using
pip intall -r requirements_2.txt 

Step 3:

dataset_dir = '/inputs'
dataset = 'busi'
input_size = 256
outputs_dir = 'busi_ukabs_run'
# Training command
python train.py --arch UKABS --dataset {dataset} --input_w {input_size} --input_h {input_size} --name {dataset}_UKABS --data_dir {dataset_dir} --output_dir {outputs_dir}


# for ACDC check the train_acdc.py and run it this way 
python train_acdc.py


This how to train the model architecture. For glas use image size 512.

Step 4: 

outputs_dir = 'busi_ukabs_run'
dataset='busi'
!python val.py --name {dataset}_UKABS --output_dir {outputs_dir}

# for ACDC
 python val_acdc.py

This is the validation code to generate the masks as well 
