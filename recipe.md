<!--
Copyright (c) 2023 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

---

version: 1.1.0

# General Variables

num_epochs: 10.0

qat_start_epoch: 0.0
qat_epochs: 3.0
qat_end_epoch: eval(qat_start_epoch + qat_epochs)

ft_epochs: 1

init_lr: 5.5e-5
min_lr: 1e-8

weight_decay: 0.00001

qat_disable_observer_epoch: eval(qat_end_epoch)
qat_freeze_bn_epoch: eval(qat_end_epoch)

distill_hardness: 0.9
distill_temperature: 5.0

# Modifiers

training_modifiers:

- !EpochRangeModifier
  start_epoch: 0
  end_epoch: eval(num_epochs)

- !SetLearningRateModifier
  start_epoch: 0.0
  learning_rate: eval(init_lr)

- !LearningRateFunctionModifier
  start_epoch: eval(qat_start_epoch)
  end_epoch: eval(qat_end_epoch)
  lr_func: cosine
  init_lr: eval(init_lr)
  final_lr: eval(min_lr)

- !LearningRateFunctionModifier
  start_epoch: eval(qat_end_epoch)
  end_epoch: eval(qat_end_epoch + ft_epochs)
  lr_func: cosine
  init_lr: eval(init_lr)
  final_lr: eval(min_lr)

- !LearningRateFunctionModifier
  start_epoch: eval(qat_end_epoch + ft_epochs)
  end_epoch: eval(qat_end_epoch + 4 \* ft_epochs)
  lr_func: cosine
  init_lr: eval(init_lr)
  final_lr: eval(min_lr)

quantization_modifiers:

- !QuantizationModifier
  start_epoch: eval(qat_start_epoch)
  ignore:
  - classifier
  - AdaptiveAvgPool2d
    disable_quantization_observer_epoch: eval(qat_end_epoch)
    freeze_bn_stats_epoch: eval(qat_end_epoch)

distillation_modifiers:

- !DistillationModifier
  hardness: eval(distill_hardness)
  temperature: eval(distill_temperature)
  distill_output_keys: [0]

regularization_modifiers:

- !SetWeightDecayModifier
  start_epoch: 0
  weight_decay: eval(weight_decay)

---

# EfficientNet_v2-s Quantized on the Imagenet dataset

This recipe defines the hyperparams necessary to quantize the Efficient_v2-s model, which was originally trained through and available via [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html). This resulting model achieves 83.47% top-1 and 96.58% top-5 accuracy on the validation set.

Users are encouraged to experiment with the hyperparams such as training length and learning rates to either expedite training or to produce a more accurate model.
This can be done by either editing the recipe or supplying the --recipe_args argument to the training commands.
For example, the following appended to the training commands will change the number of epochs and the initial learning rate:

```bash
--recipe_args '{"num_epochs":8,"init_lr":0.0001}'
```

## Training

To set up the training environment, [install SparseML with PyTorch](https://github.com/neuralmagic/sparseml#installation)

The following command was used to quantize the Efficient_v2-s model on the ImageNet dataset using four GPUs A100.

```bash
python -m torch.distributed.run --nproc_per_node 4 --no_python sparseml.image_classification.train \
    --checkpoint-path zoo:cv/classification/efficientnet_v2-s/pytorch/sparseml/imagenet/base-none \
    --arch-key efficientnet_v2_s \
    --output-dir quantized_efficientnet_v2_s \
    --recipe zoo:cv/classification/efficientnet_v2-s/pytorch/sparseml/imagenet/base_quant-none \
    --teacher-arch-key efficientnet_v2_s \
    --distill-teacher zoo:cv/classification/efficientnet_v2-s/pytorch/sparseml/imagenet/base-none \
    --pretrained-teacher-dataset imagenet \
    --dataset-path /PATH/TO/DATASET \
    --opt rmsprop \
    --batch-size 32 \
    --gradient-accum-steps 5 \
    --auto-augment imagenet \
    --random-erase 0.1 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --cutmix-alpha 1.0 \
    --norm-weight-decay 0.0 \
    --train-crop-size 300 \
    --model-ema \
    --val-crop-size 384 \
    --val-resize-size 384 \
    --ra-sampler \
    --ra-reps 4 \
    --workers 16 \
    --logging_steps 200 \
    --eval_steps 400
```

## Evaluation

This model achieves 83.47% top-1 and 96.58% top-5 accuracy on the validation set. The following command can be used to verify accuracy.

```bash
sparseml.image_classification.train \
    --checkpoint-path zoo:cv/classification/efficientnet_v2-s/pytorch/sparseml/imagenet/base_quant-none \
    --arch-key efficientnet_v2_s \
    --test-only \
    --dataset-path /PATH/TO/IMAGENET \
    --val-crop-size 384 \
    --val-resize-size 384 \
    --batch-size 64
```
