<div align="center">
    <h1>AgriIntel : AI-POWERED SMART FARMING ASSISTANT</h1>
</div>

![Screenshot 2025-01-22 210034](https://github.com/user-attachments/assets/5b945e7d-bbb0-4463-b06f-681445e102bd)

## Overview

Smart Farming Assistant is an innovative solution that uses advanced technology to help farmers improve productivity and make better decisions. The platform features a Smart Crop Recommendation system powered by machine learning to suggest optimal crops based on soil nutrients, climate, and historical data. It also includes a Plant Disease Identification tool using convolutional neural networks (CNNs) to accurately diagnose plant diseases from uploaded images, enabling timely intervention. Additional features such as real-time Weather Forecasts, tailored Fertilizer Recommendations based on soil quality and crop requirements, and a Smart Farming Guide for crop management further enhance its value. With a user-friendly web app, farmers can easily access these insights and tools to improve farming practices.

## Research Paper

> This project is based on the research paper published on IEEE. You can find the paper at the following link:

- [IEEE : Smart Crop Recommendation System with Plant Disease Identification](https://ieeexplore.ieee.org/document/10738975)

- You can view the research paper directly here : [View Paper](https://github.com/ravikant-diwakar/AgriSens/blob/master/IEEE_Paper_Smart_Crop_Recommendation_System_with_Plant_Disease_Identification.pdf)


## Features

- [x] **Smart Crop Recommendation**: Leverages machine learning to suggest the most suitable crops based on soil nutrients, climate, and historical data.
- [x] **Plant Disease Identification**: Uses convolutional neural networks (CNNs) to accurately detect and classify plant diseases from uploaded images, allowing for timely interventions.
- [x] **Fertilizer Recommendation**: Offers customized fertilizer recommendations based on soil quality and crop needs to optimize growth and yield.
- [x] **Today's Weather Forecast**: Delivers real-time weather updates, including temperature and humidity, to help farmers plan their activities effectively.
- [x] **Smart Farming Guide**: Provides guidance on crop planting schedules and management strategies to maximize productivity based on soil and weather conditions.
- [x] **User-Friendly Interface**: Features an intuitive platform for farmers to easily input data and receive personalized crop, disease, and fertilizer recommendations.


## Datasets

The **Smart Farming Assistant** project provides three key datasets: the **Crop Recommendation Dataset** (2200 rows) includes soil and environmental factors such as nitrogen, phosphorous, temperature, humidity, and pH to predict the most suitable crops; the **Plant Disease Identification Dataset** contains 70,295 training and 17,572 validation images covering 38 diseases across 14 crops like Apple, Tomato, and Grape, used to train CNN models for disease detection; and the **Fertilizer Recommendation Dataset** offers data on soil quality and crop needs to provide tailored fertilizer suggestions. These datasets can be accessed via the following links: [Crop Recommendation Dataset](https://github.com/ravikant-diwakar/AgriSens/blob/master/Datasets/Crop_recommendation.csv), [Plant Disease Dataset](https://github.com/ravikant-diwakar/AgriSens/tree/master/Datasets), and [Fertilizer Recommendation Dataset](https://github.com/ravikant-diwakar/AgriSens/blob/master/Datasets/Fertilizer_recommendation.csv).

# 📌 Crop Recommendation Model

The **Crop Recommendation Model** utilizes machine learning algorithms to suggest the most suitable crops for farmers based on environmental and soil factors. By analyzing data such as soil nutrients, temperature, humidity, pH, and rainfall, the model provides tailored crop recommendations to ensure optimal growth and productivity. The model uses seven classification algorithms, with **Random Forest** achieving the highest accuracy of 99.55%. This helps farmers make informed decisions on crop selection, ensuring better yields and efficient farming practices.

## Dataset

This dataset consists of **2200 rows** in total.
**Each row has 8 columns representing Nitrogen, Phosphorous, Potassium, Temperature, Humidity, PH, Rainfall and Label**
NPK(Nitrogen, Phosphorous and Potassium) values represent the NPK values in the soil. Temperature, humidity and rainfall are the average values of the sorroundings environment respectively. PH is the PH value present in the soil. **The Label column tells us the type of crop that's best suited to grow based on these conditions.
Label is the value we will be predicting**


## Model Architecture
 
For the Crop Recommendation Model, seven classification algorithms were utilized to predict suitable crop recommendations. These algorithms include:

- Decision Tree
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest (achieved the best accuracy)
- XGBoost
- KNN
  
Each algorithm was trained on a dataset comprising various factors such as soil nutrients, climate conditions, and historical data to provide accurate crop recommendations to farmers.

## Integration

These two models are integrated into the Smart Crop Recommendation System with Plant Disease Identification. This system provides farmers with comprehensive support, offering both crop recommendations based on various factors and precise identification of crop diseases through image analysis. By combining these models, the system enables farmers to make informed decisions, optimize crop selection, and effectively manage plant diseases for sustainable agriculture and enhanced productivity.

## System Architecture


> For a visual overview of the architecture, refer to the diagram below:

<details>

<summary>💻 System Architecture</summary>

### System Architecture

![20250124_135249](https://github.com/user-attachments/assets/1c660b6b-5b70-440e-a453-bf802b490bdc)

</details>

## Results

- Seven classification algorithms were evaluated for crop recommendation tasks.
- The accuracy of each algorithm was assessed, with the Random Forest algorithm achieving the highest accuracy of 99.55%.
- Table 1 below illustrates the accuracy achieved by each algorithm:

> [!IMPORTANT]
> The Random Forest algorithm achieved the highest accuracy of 99.55% in crop recommendation, making it the most reliable model for this system.

**Table 1: Accuracy vs Algorithms**

| Algorithm            | Accuracy   |
| --- | :---: |
| Decision Tree        | 90.0       |
| Gaussian Naive Bayes| 99.09      |
| Support Vector Machine (SVM) | 10.68 |
| Logistic Regression  | 95.23      |
| Random Forest        | 99.55      |
| XGBoost              | 99.09      |
| KNN                  | 97.5       |




# 📌Plant Disease Identification Model 

The **Plant Disease Identification Model** utilizes Convolutional Neural Networks (CNN) to accurately identify plant diseases from leaf images. Trained on the **Plant Disease Image Dataset**, which includes 70,295 images in the training set and 17,572 images in the validation set, the model covers 38 different plant disease classes across 14 crops. It detects and classifies diseases such as **Apple Scab**, **Tomato Blight**, and **Powdery Mildew**, offering farmers a reliable tool for early disease detection. 

## Dataset

The **Plant Disease Image Dataset**, used for crop disease identification, consists of 70,295 plant images from the training set and 17,572 images from the validation set, covering a variety of 38 different plant disease classes. The images are standardized to a resolution of 128x128 pixels, and the dataset occupies approximately five gigabytes of storage space.


## Model Architecture
   
For the Plant Disease Identification Model, a Convolutional Neural Network (CNN) architecture was employed. This CNN model was specifically trained for crop disease identification. Leveraging deep learning techniques, the CNN analyzes images of plant leaves to detect and classify diseases accurately. This model aids farmers in early disease detection and management, contributing to improved crop health and yield.


### Key Features:
- **Crop Specific**: The model is designed to diagnose diseases for a specific set of crops.
- **Disease Diagnosis**: It can classify diseases based on images of leaves.
- **Accuracy**: The CNN model demonstrates high accuracy in identifying plant diseases, helping farmers and researchers detect issues early.

### Supported Crops and Diseases:
- The model works with a predefined list of 14 crops.
- For each crop, the model is trained to detect and classify up to 38 specific diseases.

### Crop Disease Guide

> Follow this link for detailed information on the Crop Disease Guide.

- [x] 📄 [Crop Disease Guide](DISEASE-GUIDE.md)

### How it Works:
- The model uses images of plant leaves to detect symptoms of various diseases.
- It applies CNN-based image classification to identify the correct disease for a given crop.


These results demonstrate the effectiveness of the Smart Crop Recommendation System with Plant Disease Identification in assisting farmers with informed crop selection and disease management, thereby contributing to improved agricultural practices and crop yields.

> [!IMPORTANT]
> ### System Requirements
>  - Python: Version 3.8 or above
>  - TensorFlow/Keras: For disease identification
>  - Streamlit: For creating the web interface

> [!TIP]
> ### Common Issues and Tips
> - Ensure all dependencies in the `requirements.txt` are installed.
> - For TensorFlow-based disease detection, ensure you have a compatible GPU or CPU for faster processing.



## 👨‍💻 CONTRIBUTERS
- [Shah Archit](https://github.com/iamarchitshah)
- [Bhalani Tirth](https://github.com/D24IT163)
- [Bhingradiya Jay](https://github.com/BJAYG12)


