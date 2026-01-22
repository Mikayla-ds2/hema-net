# hema-net : a deep learning classifier for blood cells
hema-net is an image classification project built with the machine learning framework, PyTorch, that identifies types of blood cells from microscopic images! It was designed to practice and refine deep learning skills, particulary working with Convulotional Neural Networks, managing large, imbalanced image datasets, and building clean, detailed, efficient ML pipelines. I trained hema-net on the BloodMNIST [https://github.com/MedMNIST/MedMNIST/tree/main] dataset and reached a final accuracy of 94%!

# Why I Built This

I'm super passionate about working in precision medicine using computational methods, and this project acted as a rudimentary introduction of a popular niche in the industry: medical ML. I really wanted to do my own version of working with medical data and in the process, I was also able to level up my PyTorch skills and develop reusable ML functions I can repurpose for future projects! I am also super excited to replicate this same project in another medical domain to identify different diseases or illnesses. I've really been able to dive deep into the process of image classification and could really explore and experiment with all factors.

# How I Built This

- Dataset: BloodMNIST from the MedMNIST suite (8 classes, a total of 17,092 images split 7:1:2 training, validation, test)
- Framework: PyTorch
- Model: CNN with multiple convolutional, pooling, and dropout layers + ReLU activation functions, and batch normalization
- Optimizer: AdamW
- Scheduler: Reduce LR on Plateau
- Criterion: Focal Loss w/ boost factors
- Early Stopping Patience: 10 epochs
- Early Stopping Mininum Delta: 0.005
- Total Parameters: 11,310,408
- Used Random Horizontal Flip, Random Rotation of 15 degrees, and Color Jitter as data augmentations
- Used confusion matrices, classification reports, and multiple plots comparing training and validation loss and accuracy, the learning rate, and training time per epoch

# What I Learned

- How to structure a PyTorch project with dataloaders, transforms, model classes, and training loops
- Best practices for image classification (e.g., augmentations, batch normalization)
- How class imbalance affects accuracy metrics

# What I Struggled With

- Getting the model to generalize well without overfitting small classes
- Understanding the balance between training longer vs tweaking architecture
- How to deal with class imbalance (L1/L2 regularization, Focal Loss etc.)

# Submission Notes

This project is part of my Athena portfolio and represents 20+ hours of hands-on work and research. It was my first deep dive into PyTorch and image classification, and I’m proud of how far it’s come.

If you’re just getting into ML or bioinformatics, I hope HEMA-Net helps you get inspired too.

[![Athena Award Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Faward.athena.hackclub.com%2Fapi%2Fbadge)](https://award.athena.hackclub.com?utm_source=readme)
