# AMS_feathers_in_focus
This repository contains the work of Group 24 | Feathers in Focus for the Applied Machine Learning course in the Data Science Master track.  
The goal of the assignment is to classify 200 bird species using a dataset of ~4,000 training images and ~4,000 test images, evaluated via a Kaggle competition.


# Repo structure
- **Models:** Saved model weights (`.pth`) for all trained models, loadable via PyTorch.
- **Notebooks:** Jupyter notebooks containing the full training and evaluation pipelines for each model. All notebooks use fixed random seeds and relative paths to support reproducibility.
- **runs:** TensorBoard logs for resnet and mobile_vgg model
- **Submission_csv:** All of the submission files for the kaggle competition are saved here in the required csv format
- **kaggle repo data:** Dataset provided by Kaggle:
    - *test_images*: a folder containing all the test images
    - *train images*: a folder containing all the train images
    - *attributes.npy*: can be used to load in attributes with numpy belonging to each class
    - *attributes.txt*: additional info about the attributes
    - *class_names.npy*: can be used to load in the class names belonging to the numerical labels
    - *test_images_path.csv*: a csv file including the correct path to test images with dummy labels
    - *test_images_sample.csv*: a given example submission for formatting purposes
    - *train_images.csv*: a csv file including the correct path to train images including correct labels
- **environment.yaml:** Conda environment specification (`.yml`) listing all required packages.

# Sources
## ResNet18 model
### Model Architecture & Transfer Learning
- **Base Architecture:** [PyTorch Tutorial: Transfer Learning for Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
    - Used as the primary reference for loading the ResNet18 backbone and modifying the final fully connected layer.

### Optimization & Training Techniques
- **Optimizer (AdamW):** [PyTorch AdamW Documentation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
  -  We selected AdamW instead of standard Adam or SGD for its decoupled weight decay implementation, which improves generalization.
- **Scheduler (Cosine Annealing):** [CosineAnnealingLR Documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
  - Used to smoothly decay the learning rate, helping the model converge to a better local minimum.
- **Regularization (Label Smoothing):** [CrossEntropyLoss Documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
  - Applied `label_smoothing=0.1` to the classification loss to prevent the model from becoming overconfident on noisy data.

### Multi-Task & Attribute Learning
- **Handling Imbalanced Attributes:** [BCEWithLogitsLoss & pos_weight](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
  - We utilized `BCEWithLogitsLoss` with a dynamically calculated `pos_weight` to penalize the model heavily for missing rare attributes (visual traits).
- **Multi-Label Concept:** [Multi-Label Image Classification with PyTorch](https://towardsdatascience.com/multi-label-image-classification-with-pytorch-image-tagging-5a41a4a441e8) 
  - Reference for structuring the secondary output head for attribute prediction.*

## Vision Transformer
- **Baseline model:** [Hugging face ViT](https://huggingface.co/google/vit-base-patch16-224)
    - Used as a baseline to google/vit-base-patch16-224

## Initial models from scratch: Inspired by VGG and Mobilenet
- **VGG:** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
    - Used as inspiration for the vgg_model
- **MobileNet:** [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
    - Used as inspiration for the vgg_mobile model
