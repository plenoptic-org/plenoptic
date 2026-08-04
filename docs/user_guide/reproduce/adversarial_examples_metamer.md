---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`adversarial_examples_metamer.ipynb`**!

Run it in your browser: **{binder}`adversarial_examples_metamer.ipynb`**!

:::

(adversarial_examples)=
# Generate adversarial examples using Metamer

:::{warning}
This notebook requires the optional dependency `torchvision`, which can be installed with `pip`.
:::

In this notebook we demonstrate how we can use the {class}`~plenoptic.Metamer` class to synthesize adversarial examples. Adversarial examples are tiny perturbations to an image that causes Deep Neural Networks to misclassify ({cite:alp}`Szegedy2013`, {cite:alp}`goodfellow_explaining_2015`).

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import torch
from myst_nb import glue

import plenoptic as po

# this notebook uses torchvision, which is an optional dependency. if this import fails,
# install torchvision in your plenoptic environment and restart the notebook kernel.
try:
    import torchvision
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "optional dependency torchvision not found!"
        " please install it in your plenoptic environment "
        "and restart the notebook kernel"
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# set seed for reproducibility
po.set_seed(4)
```

## Prepare model and image for synthesis

In the following block, we create a {class}`~plenoptic.models.DeepNetFeatures` model matching the output of the final fully connected layer of ResNet50. After creating the model, we then prepare the image. Finally, we ensure that the model and image have the proper device and dtype, and remove the gradient from all model parameters.

To learn more about any of these steps and why we take them, read [](deep_nets).

```{code-cell} ipython3
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
deepnet = torchvision.models.resnet50(weights=weights)
deepnet.eval()
transform = weights.transforms()
norm = torchvision.transforms.Normalize(transform.mean, transform.std)
target_layer = "layer4"
model = po.models.DeepNetFeatures(deepnet, target_layer, norm)

img = po.data.macaque()
img = po.process.blur_downsample(img, 2)[..., :-59, :]
img = po.process.center_crop(img, transform.crop_size[0])
po.plot.imshow(img, as_rgb=True)
img = img.to(DEVICE).to(torch.float64)
model.to(DEVICE).to(torch.float64)
deepnet.to(DEVICE).to(torch.float64)
po.remove_grad(model)
```

## Prepare the target image

```{code-cell} ipython3
full_dataset = torchvision.datasets.Caltech256(
    root="/Users/raffleszhu/Documents/GitHub/notebooks_plenoptic/data", download=True
)
target_img, _ = full_dataset[10262]
target_img = torchvision.transforms.ToTensor()(target_img)
target_img = torchvision.transforms.functional.resize(
    target_img, [img.shape[-2], img.shape[-1]], antialias=True
)
target_img = target_img.unsqueeze(0)
po.plot.imshow(target_img, as_rgb=True)
target_img = target_img.to(DEVICE).to(torch.float64)
```

## Visualizing classification of the clean image and target image
First let us extract all the ImageNet categories:

```{code-cell} ipython3
imagenet_categories = np.asarray(weights.meta["categories"])
```

Let us define two helper functions:
- `convert_to_probs` converts the activation in the final fully-connected layer to probabilities that sum to 1. It gets used in `get_category`, below.
- `get_category` accepts a single image and returns both a vector containing the category probabilities and name of the category with the highest probability.

```{code-cell} ipython3
def convert_to_probs(logits):
    return torch.nn.functional.softmax(logits, dim=1).squeeze()


def get_category(image):
    category_probs = convert_to_probs(deepnet(norm(image))).detach().cpu()
    category = imagenet_categories[category_probs.argmax()]
    return category_probs, category
```

ResNet50 is trained to classify images into one of [1000 categories](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). The following plot shows the classification probabilities for our initial image as a stem plot. Each of the 1000 categories is represented by a line, whose y-value gives the model's probability that the image belongs to the corresponding category (the x-value is arbitrary). The title shows the label of the most likely category, and the text on the plot shows the other categories with probability higher than 0.01.

```{code-cell} ipython3
category_probs, category = get_category(img)
po.plot.stem_plot(category_probs, title=category)
likely_cats = "\n- ".join(list(imagenet_categories[category_probs > 0.01]))
plt.text(700, 0.5, f"Likely categories:\n- {likely_cats}");
```

The category of our initial image, [guenon](https://en.wikipedia.org/wiki/Guenon), is an Old World monkey. Though it isn't the actual species of the monkey in question (a [Celebes crested macaque](https://en.wikipedia.org/wiki/Celebes_crested_macaque)), it's a reasonable category for it. Notice the model is highly confident in its classification, with a probability of about 0.8 and no other category exceeding a probability of 0.1. The other predicted categories are all [Old World monkeys](https://en.wikipedia.org/wiki/Old_World_monkey).

```{code-cell} ipython3
category_probs, category = get_category(target_img)
po.plot.stem_plot(category_probs, title=category)
likely_cats = "\n- ".join(list(imagenet_categories[category_probs > 0.01]))
plt.text(700, 0.5, f"Likely categories:\n- {likely_cats}");
```

## Define a penalty

To qualify as an adversarial example, the image must satisfy two requirements: (1) the perturbation in image space is small and (2) the model outputs an incorrect classification with high confidence ({cite:alp}`goodfellow_explaining_2015`).

```{code-cell} ipython3
def custom_penalty(image):
    epsilon_penalty = po.metric.mse(image, img).mean()
    inside_penalty = po.regularize.penalize_range(image, allowed_range=(0, 1))
    return epsilon_penalty + inside_penalty
```

## Synthesize the adversarial image

```{code-cell} ipython3
penalty_lambda = 1000
met = po.Metamer(
    target_img, model, penalty_function=custom_penalty, penalty_lambda=penalty_lambda
)
```

:::{admonition} How does {attr}`~plenoptic.MADCompetition.metric_tradeoff_lambda` affect the adversarial image
:class: dropdown hint

If you are applying this procedure to new images or new image classification models, you will almost certainly need to experiment to find the appropriate {attr}`~plenoptic.MADCompetition.metric_tradeoff_lambda`

:::

```{code-cell} ipython3
met.setup(initial_image=img, optimizer_kwargs={"lr": 0.001})
```

```{code-cell} ipython3
met.synthesize(store_progress=True, max_iter=10)
po.plot.synthesis_status(met);
```

## Visualizing the adversarial image

First let us visualize the category probabilities of the original, target, and advesarial images using stem plots.

```{code-cell} ipython3
images = {"Original": img, "Target": target_img, "Adversarial": met.metamer}
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
for i, name in enumerate(["Original", "Target", "Adversarial"]):
    category_probs, category = get_category(images[name])
    likely_cats = "\n- ".join(list(imagenet_categories[category_probs > 0.05]))
    most_likely_cat = imagenet_categories[category_probs.argmax()]
    mse_val = po.metric.mse(images["Original"], images[name]).mean().item()
    title = (
        f"{name}: {most_likely_cat} (p={category_probs.max():.2f})\nMSE={mse_val:.2e}"
    )
    po.plot.imshow(
        images[name],
        ax=axes[i, 0],
        as_rgb=True,
        title=title,
    )
    po.plot.stem_plot(category_probs, ax=axes[i, 1], ylim=False)
    axes[i, 1].set_title("Categories")
    axes[i, 0].xaxis.set_visible(False)
    axes[i, 0].yaxis.set_visible(False)
    axes[i, 1].text(
        1, 0.5, f"Likely categories:\n- {likely_cats}", transform=axes[i, 1].transAxes
    )
category_probs, category = get_category(met.metamer)
glue("category_name", str(category), display=False)
```

```{code-cell} ipython3
mse = po.metric.mse(img, met.metamer)
title = f"Adversarial - Original \nMSE={mse_val:.2e}"
diff = (
    met.metamer - img + 1
) / 2  # convert the range from [-1,1] to [0,1] for RGB images
po.plot.imshow(diff, as_rgb=True, title=title, col_wrap=2, vrange="auto0");
```

```{code-cell} ipython3
channelwise_diffs_met = met.metamer - img
titles = [
    "Adversarial - Original (R)",
    "Adversarial - Original (G)",
    "Adversarial - Original (B)",
]
po.plot.imshow(channelwise_diffs_met, col_wrap=3, title=titles, vrange="auto0");
```

This notebook demonstrates how to generate adversarial examples using the {class}`~plenoptic.Metamer` class. We encourage you to experiment with different image classification networks, images, and hyperparameters to generate other adversarial examples yourself!
