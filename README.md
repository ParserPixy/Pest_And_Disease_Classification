# Pest_And_Disease_Classification

Computer Vision Task to detect and classify images on the CCMT Dataset

## Dataset

- [CCMT Plant Disease Dataset](https://doi.org/10.1016/j.dib.2023.109306), which consists of plant leaves, pests, fruits and images of sick parts of cashew, cassava, maize, and tomato.
- The dataset is comprehensive and consists of 102,976 high-quality images of four crops with 22 different classes, respectively cashew (5 classes), cassava (5 classes), maize (7 classes), and tomato (5 classes).

## Model

- Pytorch's architecture of [EfficientV2 model](https://arxiv.org/abs/2104.00298)
- An INT8 quantized version of the [EfficientNet_v2-s model trained on the ImageNet dataset](https://sparsezoo.neuralmagic.com/models/efficientnet_v2-s-imagenet-base_quantized?hardware=deepsparse-c6i.12xlarge&comparison=efficientnet_v2-s-imagenet-base) is used for speeding up training.

## Workflow

- Creating your recipe for model sparsification (Refer [Recipe Documentation](https://github.com/neuralmagic/sparseml/blob/main/docs/source/recipes.md))
- The sparsified model is fine-tuned on the CCMT Dataset over 10 epochs and using the `recipe.md` and [SparseML library](https://github.com/neuralmagic/sparseml).
- The model is exported to ONNX format for inference with [DeepSparse](https://github.com/neuralmagic/deepsparse/blob/main/docs/user-guide/deepsparse-pipelines.md)

## Results

| Training                                         | Validation                                            |
| ------------------------------------------------ | ----------------------------------------------------- |
| ![Train Accuracy](/images/train_Accuracy.jpeg)   | ![Validation Accuracy](/images/valid_Accuracy.jpeg)   |
| ![Train Loss](/images/train_loss.jpeg)           | ![Validation Loss](/images/valid_loss.jpeg)           |
| ![Train Precision](/images/train_Precision.jpeg) | ![Validation Precision](/images/valid_Precision.jpeg) |
| ![Train Recall](/images/train_Recall.jpeg)       | ![Validation Recall](/images/valid_Recall.jpeg)       |
| ![Train AUC Score](/images/train_AUC_Score.jpeg) | ![Validation AUC Score](/images/valid_AUC_Score.jpeg) |
| ![Train F1 Score](/images/train_F1_Score.jpeg)   | ![Validation F1 Score](/images/valid_F1_Score.jpeg)   |
