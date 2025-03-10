# Skull Stripping and Registration of Brain Images
This project performs skull stripping and affine registration of brain images using the MNI152 template and brain mask. The core steps of the process involve registering an input brain image to an atlas template, applying affine transformations, and performing skull stripping using a brain mask.

## How It Works
**Image Registration:** The input brain image is registered to the MNI152 template using affine transformations to align the images.

**Skull Stripping:** After registration, a brain mask is applied to strip the skull from the transformed image. The process uses the MNI152 brain mask and resamples it to match the transformed image's shape and affine matrix.

**Requirements**

Python 3.11+

**Required libraries:**

Please look at Poetry.lock file for the required libraries.

**To run this project, follow these steps:**

1. Create a Virtual Environment
Create a virtual environment to keep your project dependencies isolated.


python -m venv o8t_venv

2. Activate the Virtual Environment
Activate the virtual environment. 

**On Windows:**

```bash
o8t_venv\Scripts\activate
```

**On macOS/Linux:**

```bash
source o8t_venv/bin/activate
```

3. Install Dependencies

Install the required dependencies using poetry.

```bash
poetry install
```

In case the above command does not work, you can install the dependencies manually using pip.
This file is created for python 3.8. 

```bash
pip install -r requirements.txt
```

4. Run the Code

```commandline
python process_mri.py --input sub-0002_ses-01_T1w.nii.gz --output output.nii.gz --sigma 1.5
```

## Use of AI Assistance in This Challenge
During this challenge, I used AI assistance (ChatGPT) primarily for structuring and refining parts of my implementation. 
Below is a breakdown:

Code Refinement & Debugging
General AI-generated solutions for registration process and skull stripping 

## Usage examples

![image](/statics/output_comparison.png)

# Description of the approach

The approach follows a structured pipeline for skull stripping and affine registration of brain MRI images using the MNI152 template. 
First, the input MRI scan is registered to the MNI152 template using an affine transformation, ensuring alignment with a standardized anatomical space. 
This is achieved using mutual information as the similarity metric and an iterative optimization process to compute the best transformation matrix. 
Once registered, the MNI152 brain mask is resampled to match the transformed image's shape and affine matrix. 
The resampled mask is used to create a binary mask, which is applied to the transformed image to remove skull. 
The final skull-stripped image is saved, preserving the brain region while eliminating surrounding tissues. 
Throughout this process, validation steps such as inspecting affine matrices and visualizing intermediate outputs ensure the accuracy of the transformations and skull stripping.

# Assumptions made

Input Image Quality and Orientation: The input brain MRI scan is assumed to be of high quality to facilitate accurate registration with the MNI152 template.

Affine Registration Sufficiency: The affine transformation is assumed to be adequate for aligning the input image with the MNI152 template. Nonlinear deformations, which might provide more precise alignment, are not considered in this implementation.

MNI152 Template Compatibility: The MNI152 template and brain mask are assumed to be appropriate references for skull stripping across different MRI scans. It is expected that the input image does not significantly deviate from the standard anatomical structure.

# Potential improvements

1. Use of Nonlinear Registration
2. Better Skull Stripping via Deep Learning
3. Refined Masking Strategy
4. Multi-Atlas Approach
5. Automated Quality Control
6. GPU acceleration
7. Handling of Partial Brain Scans
8. Add User-Defined Parameters
9. Support for Other MRI Modalities