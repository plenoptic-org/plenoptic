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

(adversarial_examples_metamer)=
# Generate adversarial examples using Metamer

:::{warning}
This notebook requires the optional dependency `torchvision`, which can be installed with `pip`.
:::

In this notebook we demonstrate how we can use the {class}`~plenoptic.Metamer` class to synthesize adversarial examples. Adversarial examples are images with subtle, imperceptible perturbations designed to deceive a Deep Neural Networks into making an incorrect classification ({cite:alp}`Szegedy2013`, {cite:alp}`goodfellow_explaining_2015`). In a non-strict sense, an adversarial example is a metamer of the new (misclassified) class because the model has the same classification behaviour for the adversarial image and other images in the class. On this basis, we can use {class}`~plenoptic.Metamer` to generate adversarial examples by matching the model output representation between the synthesized image and another image in that class. For an in-depth dive of creating metamers of Deep Neural Networks, read [](feather_2023) and [](deep_nets).

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

## Prepare model and images for synthesis

In the following block, we create a {class}`~plenoptic.models.DeepNetFeatures` model matching the output of `"layer4"` of a classic image recognition network, [ResNet50](https://en.wikipedia.org/wiki/Residual_neural_network), trained to classify images into one of [1000 categories](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/).. After creating the model, we then prepare the original image and an image of a different class whose represetnation we try to match. Finally, we ensure that the model and images have the proper device and dtype, and remove the gradient from all model parameters.

To learn more about any of these steps and why we take them, read [](deep_nets).

As part of this procedure, we must choose an image whose `layer4` representation (and thus, categorization) we would like to match. For this example let us fool the network into thinking a macaque is a cheeseburger (image taken from the Caltech256 dataset {cite:alp}`griffin_caltech_2022`).

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

target_img = po.load_images(po.data.fetch_data("caltech256_burger.jpg"), as_gray=False)
target_img = po.process.center_crop(target_img, img.shape[-1])
po.plot.imshow(target_img, as_rgb=True)

img = img.to(DEVICE).to(torch.float64)  # convert to float64 for reproducibility
target_img = target_img.to(DEVICE).to(torch.float64)
model.to(DEVICE).to(torch.float64)
deepnet.to(DEVICE).to(torch.float64)

# remove the gradient from all model parameters
po.remove_grad(model)
```

## Visualizing classification of the clean image and target image
First let us extract all the [ImageNet-1K](https://en.wikipedia.org/wiki/ImageNet#ImageNet-1K) categories:

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

ResNet50 is trained to classify images into one of [1000 categories](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). The following plot shows the classification probabilities for the original image as a stem plot. Each of the 1000 categories is represented by a line, whose y-value gives the model's probability that the image belongs to the corresponding category (the x-value is arbitrary). The title shows the label of the most likely category, and the text on the plot shows the other categories with probability higher than 0.01.

```{code-cell} ipython3
category_probs, category = get_category(img)
po.plot.stem_plot(category_probs, title=category)
likely_cats = "\n- ".join(list(imagenet_categories[category_probs > 0.01]))
plt.text(700, 0.5, f"Likely categories:\n- {likely_cats}");
```

The category of our initial image, [guenon](https://en.wikipedia.org/wiki/Guenon), is an Old World monkey. Though it isn't the actual species of the monkey in question (a [Celebes crested macaque](https://en.wikipedia.org/wiki/Celebes_crested_macaque)), it's a reasonable category for it. Notice the model is highly confident in its classification, with a probability of about 0.8 and no other category exceeding a probability of 0.1. The other predicted categories are all [Old World monkeys](https://en.wikipedia.org/wiki/Old_World_monkey). Now let us make the same plot for the target image.

```{code-cell} ipython3
category_probs, category = get_category(target_img)
po.plot.stem_plot(category_probs, title=category)
likely_cats = "\n- ".join(list(imagenet_categories[category_probs > 0.01]))
plt.text(700, 0.5, f"Likely categories:\n- {likely_cats}");
```

The category of the target image is cheeseburger. This is the desired category of our adversarial example.

+++

## Synthesize the adversarial image

To qualify as an adversarial example, the image must satisfy two requirements ({cite:alp}`goodfellow_explaining_2015`): 
1. The perturbation in image space is small.
2. The model outputs an incorrect classification with high confidence.

By starting with the original image and changing pixel values until the model output representation is matched between the synthesized and target images, the model should misclassify the synthesized image. However, we also need to constrain the synthesized image to be close to the original image in pixel space. To do this we define a penalty function that calculates the Mean Squared Error (MSE) between the original and synthesized images. In the penalty function we also add another term that penalizes pixels in the synthesized image whose values values outside $[0, 1]$.

```{code-cell} ipython3
def custom_penalty(image):
    # compute MSE independently per RGB channel and then average.
    epsilon_penalty = po.metric.mse(image, img).mean()
    inside_penalty = po.regularize.penalize_range(image, allowed_range=(0, 1))
    return epsilon_penalty + inside_penalty
```

The relative weight of the penalty in the objective function is controlled by the {attr}`~plenoptic.Metamer.penalty_lambda`. We found a value of 1000 generally worked well in our experiments.

```{code-cell} ipython3
met = po.Metamer(
    target_img, model, penalty_function=custom_penalty, penalty_lambda=1000
)
```

:::{admonition} How does {attr}`~plenoptic.Metamer.penalty_lambda` affect the adversarial image?
:class: dropdown hint

The objective function for metamer synthesis is made of two parts: the synthesis loss that measures the difference in representation between synthesized and target images, and the penalty. As {attr}`~plenoptic.Metamer.penalty_lambda` increases, the relative weight of the penalty in the objevtive increases. This helps ensure the synthesized and original images are close to each other in pixel space, at the cost of making representation matching more difficult. If you are applying this procedure to new images or new image classification models, you will almost certainly need to experiment to find the appropriate {attr}`~plenoptic.Metamer.penalty_lambda`.

:::

+++

We also decreased the learning rate from the default, as this resulted in better solutions in our experiments.

```{code-cell} ipython3
met.setup(initial_image=img, optimizer_kwargs={"lr": 0.001})
```

We let the optimization run until loss converges by setting `max_iter` to a large value.

```{code-cell} ipython3
met.synthesize(store_progress=True, max_iter=10000)
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

We see the network is highly confident that the synthesized image is a {glue}`category_name`!

+++

While the synthesized image is visually similar to the original image (also note the low MSE reference metric loss), let us verify more rigorously. To do this we subtract the the original image from the synthesized image to visualize the changes in pixel values.

```{code-cell} ipython3
mse = po.metric.mse(img, met.metamer)
title = f"Adversarial - Original \nMSE={mse_val:.2e}"
diff = (
    met.metamer - img + 1
) / 2  # convert the range from [-1,1] to [0,1] for RGB images
po.plot.imshow(diff, as_rgb=True, title=title, col_wrap=2, vrange="auto0");
```

The difference between synthesized and original images looks like random pixel noise distributed across the image. However, the noise is too faint for us to tell if there's any structure. Instead, let us try visualizing the difference in each color channel (red, green, and blue) separately.

```{code-cell} ipython3
channelwise_diffs_met = met.metamer - img
titles = [
    "Adversarial - Original (R)",
    "Adversarial - Original (G)",
    "Adversarial - Original (B)",
]
po.plot.imshow(channelwise_diffs_met, col_wrap=3, title=titles, vrange="auto0");
```

We see the difference does not look like a {glue}`category_name`, or similar. At this point we can safely conclude the synthesized image is an adversarial example of the network.

+++

This notebook demonstrates how to generate adversarial examples using the {class}`~plenoptic.Metamer` class. We encourage you to experiment with different image classification networks, images, and hyperparameters to generate other adversarial examples yourself!
