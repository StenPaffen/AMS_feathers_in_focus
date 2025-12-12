# AMS_feathers_in_focus
The repository  for the AML kaggle competition of group 24. Feathers in focus

# AMS_feathers_in_focus
The repository for the AML Kaggle competition of group 24. Feathers in focus.

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
