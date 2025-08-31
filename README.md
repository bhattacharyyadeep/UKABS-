Hybrid Kolmogorov-Arnold Networks for Medical Image Segmentation


Medical image segmentation plays a vital role in diagnosis and treatment planning, but remains challenging due to the inherent complexity and variability of medical images, especially in capturing non-linear relationships within the data. We propose U-KABS, a novel hybrid framework that integrates the expressive power of Kolmogorov-Arnold Networks (KANs) with a U-shaped encoder-decoder architecture to enhance segmentation performance. The U-KABS model combines the convolutional and squeeze-and-excitation stage, which enhances channel-wise feature representations, and the KAN Bernstein Spline (KABS) stage, which employs learnable activation functions based on Bernstein polynomials and B-splines. This hybrid design leverages the global smoothness of Bernstein polynomials and the local adaptability of B-splines, enabling the model to effectively capture both broad contextual trends and fine-grained patterns critical for delineating complex structures in medical images. Skip connections between encoder and decoder layers support effective multi-scale feature fusion and preserve spatial details. Evaluated across diverse medical imaging benchmark datasets, U-KABS demonstrates superior performance compared to strong baselines, particularly in segmenting complex anatomical structures.



read the scripts files to run the code.
