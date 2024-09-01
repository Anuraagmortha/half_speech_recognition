# Half Speech Recognition
It is a Speech Recognition System for Dysartric patients. This project demonstrates an audio command recognition system using convolutional neural networks (CNNs). The system processes audio files, extracts spectrogram features, and classifies them into predefined categories.

## Getting Started


1. **Clone the Repository**

   ```
   git clone https://github.com/YourUsername/your_repository_name.git
   cd your_repository_name
   ```

2. **Set Up a Virtual Environment**

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```
   pip install -r requirements.txt
   ```

# Dataset Overview:  
The dataset used for this project is filtered from the Toronto University's TORGO Database.  
The filtered dataset consists of 10 words with 700 samples each, making a total of 7000 audio and prompt samples on which the model is trained on.  

    
![image](https://github.com/user-attachments/assets/6e9b45ea-0f4e-497b-a754-38e386eaf5c9)


## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:
  - Convolutional layers with Batch Normalization and MaxPooling.
  - Flatten layer followed by Dense and Dropout layers.
  - The final Dense layer uses a softmax activation function for classification.

The model is compiled with the Adam optimizer and uses categorical crossentropy as the loss function. It tracks accuracy as the evaluation metric.  


![image](https://github.com/user-attachments/assets/a0e3f151-dacd-472a-a18c-d2641d2bff60)  
![image](https://github.com/user-attachments/assets/bc38e688-c6f3-430a-a41b-85c319fffb74)  



## Hyperparameter Tuning

Hyperparameter tuning is performed using `RandomizedSearchCV` with the following parameters:

- **Optimizer**: Choices include 'adam' and 'rmsprop'.
- **Dropout Rate**: Choices include 0.2, 0.3, 0.4, and 0.5.

## Training and Evaluation

The model is trained using the following settings:

- **Initial Learning Rate**: `1e-3`
- **Epochs**: `100`
- **Batch Size**: `64`

The best model is saved using `ModelCheckpoint` and training is monitored with `EarlyStopping` to prevent overfitting.

# Results  

This is the Confusion matrix of the predictions:  
  
![image](https://github.com/user-attachments/assets/55db8106-d99a-452e-bcbe-e300ac7d6ca9)  


## Next Steps

1. **Expand Dataset**:
   - Increase the diversity and volume of audio samples to improve model generalization and robustness. Consider adding more commands or variations of existing commands.

2. **Experiment with Different Architectures**:
   - Explore alternative neural network architectures, such as deeper CNNs or hybrid models combining CNNs with Recurrent Neural Networks (RNNs) for better sequence modeling.

3. **Fine-Tune Hyperparameters**:
   - Perform additional hyperparameter tuning using techniques like Grid Search or Bayesian Optimization to further enhance model performance.

4. **Implement for Sentences Audio Processing**:
   - Adapt the model for real-time audio classification by integrating it with a live audio feed. This involves optimizing inference speed and handling varying audio input leng

By addressing these next steps, you can further develop and refine your audio command recognition system, making it more robust and applicable to real-world scenarios.
