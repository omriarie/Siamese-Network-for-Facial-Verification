# Siamese Network for Facial Verification

This project demonstrates the use of a Siamese neural network to perform facial verification. By leveraging **PyTorch** and other key libraries, I implemented, trained, and fine-tuned models to compare two images and predict if they belong to the same individual. Each experiment builds upon the last, showcasing progressive improvements in performance.

---

## Tools and Libraries
- **PyTorch**: For designing and training the Siamese network.
- **NumPy** and **Pandas**: For data manipulation and processing.
- **Matplotlib**: For visualizing metrics and results.
- **Torchvision**: For transformations and data augmentation.
- **Google Colab**: For GPU-accelerated training.
- **Python**: The core programming language.

---

## Dataset
- **Source**: Grayscale facial images of 5749 individuals, with ~4400 images in total.
- **Preprocessing**: 
  - Resized to `105x105` pixels.
  - Normalized and converted to PyTorch tensors.
  - Augmentation techniques included random horizontal flips, rotations, and brightness/contrast adjustments.

---

## Model Architecture
The Siamese network was iteratively enhanced across multiple experiments. Each version followed this structure:

### **Experiment 1**: Base Model
- **Convolutional Layers**:
  - Conv1: 64 filters, 10×10 kernel, stride 1, ReLU activation, and max-pooling (2×2).
  - Conv2: 128 filters, 7×7 kernel, stride 1, ReLU activation, and max-pooling (2×2).
  - Conv3: 128 filters, 4×4 kernel, stride 1, ReLU activation, and max-pooling (2×2).
  - Conv4: 256 filters, 4×4 kernel, stride 1, ReLU activation.
- **Fully Connected Layers**:
  - FC1: Input size `9216`, output size `4096`, followed by Sigmoid activation.
  - Output Layer: Input size `4096`, output size `1`, followed by Sigmoid.
- **Loss Function**: Binary Cross-Entropy Loss (BCELoss).
- **Optimizer**: Adam.
- **Test Accuracy**: **66.00%**

### **Experiment 2**: Batch Normalization and Augmentation
- **Enhancements**:
  - Added BatchNorm layers after each convolution.
  - Data augmentation: random flips, rotations, and brightness/contrast adjustments.
- **Key Changes**:
  - FC1 activation changed to ReLU.
- **Test Accuracy**: **75.00%**

### **Experiment 3**: Dropout and Learning Rate Scheduler
- **Enhancements**:
  - Added Dropout (p=0.5) to the fully connected layers to reduce overfitting.
  - Implemented StepLR scheduler to decay the learning rate.
- **Test Accuracy**: **52.90%**
- **Observations**: Excessive regularization led to underfitting.

### **Experiment 4**: Early Stopping and MSE Loss
- **Enhancements**:
  - Added early stopping with a patience of 5 epochs.
  - Switched from BCELoss to Mean Squared Error Loss (MSELoss) for smoother gradient updates.
- **Test Accuracy**: **72.60%**
- **Observations**: Early stopping prevented overfitting while maintaining stable performance.

---

## Hyperparameter Tuning
A grid search was performed to optimize the following parameters:
- **Epochs**: `10`, `20`, `30`
- **Batch Size**: `64`, `128`, `256`
- **Learning Rate**: `0.001`, `0.0001`, `0.00001`
- **Early Stop Patience**: `3`, `5`, `7`

### Best Configuration:
- Epochs: `20`
- Batch Size: `64`
- Learning Rate: `0.0001`
- Early Stop Patience: `3`

### Final Model Performance:
- **Test Loss**: **0.545**
- **Test Accuracy**: **72.80%**

---

## Key Takeaways
1. **Tool Mastery**: This project demonstrates full control over PyTorch's ecosystem, including dataloaders, loss functions, optimizers, and learning rate schedulers.
2. **Progressive Improvement**: Each experiment builds upon the previous, introducing targeted changes and assessing their impact.
3. **Challenges**:
   - Balancing regularization to avoid underfitting.
   - Addressing edge cases like occlusions and varied angles.

---

## Project Files
- **Notebook**: `ass2_notebook.ipynb` - Contains all experiments, training, and testing workflows.
- **Report**: `ass2_report.pdf` - Detailed experimental results and conclusions.

---
## Contact
For any inquiries or issues, please open an issue on the GitHub repository or contact the maintainers directly:

Omri Arie – omriarie@gmail.com  
Project Link: https://github.com/omriarie/Siamese-Network-for-Facial-Verification

