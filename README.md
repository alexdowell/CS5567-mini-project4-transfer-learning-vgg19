# CS5567 Mini Project 4: Transfer Learning with VGG19  

## Description  
This repository contains **Mini Project 4** for **CS5567**, where transfer learning is applied using the **VGG19 convolutional neural network (CNN)** for image classification. The project utilizes a dataset stored in the `archive` folder and implements transfer learning techniques to improve classification accuracy. The model is evaluated based on cosine similarity, ROC curve analysis, and AUC performance.  

## Files Included  

### **Project Code**  
- **File:** CS5567_miniProject4.m  
  - Implements transfer learning with VGG19  
  - Preprocesses and augments the dataset  
  - Trains a modified VGG19 network on 70% of the dataset  
  - Evaluates performance using cosine similarity and an ROC curve  

### **Project Report**  
- **File:** CS5567_miniProject4_ results.pptx  
  - Detailed explanation of the process  
  - Analysis of model performance and results  
  - Discussion on genuine vs. impostor classification  

### **Pre-trained Model Weights**  
- **File:** netTransfer.mat (not included due to size limitations, but will be available in the repository)  
  - Stores the trained model weights from transfer learning on VGG19  

### **Dataset (Not Included in Repository)**  
- The dataset (`archive.zip`) is used for training and validation. Due to size constraints, it is not included in the repository but can be downloaded from Kaggle or another source.  

## Installation  
Ensure **MATLAB** and the **Deep Learning Toolbox** are installed before running the script.  

### Required MATLAB Add-ons  
- **Deep Learning Toolbox**  
- **Pretrained VGG19 Network**  
  - Available in MATLAB: Run `vgg19` in the command window to check availability  

## Usage  
1. Extract `archive.zip` and place the dataset inside the repository directory  
2. Open MATLAB and run `CS5567_miniProject4.m`  
3. The script will:  
   - Load and preprocess images  
   - Train the modified VGG19 network  
   - Compute cosine similarity for authentication  
   - Generate performance metrics including histograms and ROC curves  

## Example Output  

- **Validation Accuracy:** 92%  
- **ROC Curve Analysis:**  
  - **AUC:** 98.5%  
  - **Decision Threshold:** 0.8818  
  - **Training Genuine Acceptance Rate (GAR):** 80.00%  
  - **Validation False Acceptance Rate (FAR):** 1.01%  
  - **Validation GAR:** 80.00%  

The model achieves high accuracy and robust classification but has minor overlap between impostor and genuine classes. Further improvements could involve fine-tuning the decision threshold.  

## Contributions  
This repository is designed for academic use in CS5567. Feel free to modify and extend the implementation.  

## License  
This project is open for educational and research use.  

---  
**Author:** Alexander Dowell  
