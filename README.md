# LEARN Network Implementation

## Overview

This repository contains the implementation of the LEARN (Learned Experts’ Assessment-based Reconstruction Network for Sparse-data CT ) for computed tomography (CT) reconstruction. The primary objective was to explore the effectiveness of iterative reconstruction methods using Deep Learning schemes when dealing with sparse data, which is common in medical imaging.

## Instructions

- Download the full-dose training image data (D45 3mm) from the 2016 Low-Dose CT Grand Challenge at https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/. Then, organize the data into two separate folders named 'Training' and 'Validation/Testing'. Allocate the data from the first eight patients to the 'Training' folder and from the last two patients to the 'Validation/Testing' folder. Place these two folders inside a master folder called 'AAPM-Mayo-CT-Challenge'
- Using conda, create the environment using a terminal:
  ```bash
  conda env create -f environment.yml
- Activate the environment
    ```bash
  conda activate fastCT
- Set the paramaters and run the file  <code>main.py</code> for training
- Set the path of the trained model and run <code>test.py</code> for testing the model under the metrics SSIM, PSNR and RMSE
- Use the file  <code>reconstruction.py</code> for reconstructing an image from an undersampled sinogram and using an already trained model

## Contributions

Contributions to this project are welcome. Please fork the repository and submit a pull request with your suggested changes.

## License
This project is licensed under the MIT License.

## Contact
For any queries or support, please contact jhernandezga@unal.edu.co.

## Overview results
![1](https://github.com/jhernandezga/CT_Reconstruction_LEARN_paper/assets/57245076/91212ffe-50de-4e2c-a628-7e74f3f0f807)
![2](https://github.com/jhernandezga/CT_Reconstruction_LEARN_paper/assets/57245076/5d29d969-df3e-4b63-9efc-f1fa3e0b67fc)
