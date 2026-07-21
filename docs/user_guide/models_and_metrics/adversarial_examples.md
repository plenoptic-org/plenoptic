---
jupyter:
  jupytext:
    custom_cell_magics: kql
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: plenoptic (3.13.12.final.0)
    language: python
    name: python3
---

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`adversarial_examples.ipynb`**!

Run it in your browser: **{binder}`adversarial_examples.ipynb`**!

:::

(adversarial_examples)=
# Using MAD to generate adversarial examples

:::{warning}
This notebook requires the optional dependency `torchvision`, which can be installed with `pip`.
:::

In this notebook we demonstrate how we can use the {class}`~plenoptic.MADCompetition` class to synthesize adversarial examples. Adversarial examples are tiny perturbations to an image that causes Deep Neural Networks to misclasify. In MAD competition, the goal is to generate a pair of images that have the same value for the reference metric but extremal values (highest and lowest) for the optimized metric. While its main goal is to falsify metrics/models of human perception, the underlying machinery can be readily used to generate adversarial examples. This is achieved by defining a reference metric in pixel space (the value of which we want to be low) an optimized metric in model response space (the value of which we want to be high).

```python
import matplotlib.pyplot as plt
import numpy as np
import torch

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

%load_ext autoreload
%autoreload 2

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# set seed for reproducibility
po.set_seed(0)
```

## Prepare model and image for synthesis

In this section, we initialize a plenoptic-compatible model using the weights from {external+torchvision:ref}`TorchVision <models>`. You may be also interested in checking out [](deep_nets) for details of choosing layer and preprocessing of the input image, and using models from {external+timm:doc}`timm <models>`.

### Initialize deep neural network and pre-trained weights

First, we download the model weights for ResNet50 trained on [ImageNet-1K](https://en.wikipedia.org/wiki/ImageNet#ImageNet-1K) and initialize the `torchvision` model.

```python
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
deepnet = torchvision.models.resnet50(weights=weights)
```

Next, we ensure that our model is in evaluation mode. Many models, including ResNet50, behave differently when in training and evaluation mode. In plenoptic, models are fixed and so we want the evaluation behavior (see [here](remove-grad-doc) for more details):

```python
deepnet.eval()
```

### Specify preprocessing


We create a separate preprocessing transform, using the specified `mean` and `std` to normalize the input image

```python
transform = weights.transforms()
norm = torchvision.transforms.Normalize(transform.mean, transform.std)
```

### Select layer

Next, we specify the layer to target. Because we want the network to misclassify the image, the most direct way to do this is by choosing final output layer.

```python
target_layer = "fc"
```

### Prepare the image

Now, let's prepare the image. The input image needs to be an RGB image with a height and width of 224 pixels. It should probably also be like those found in ImageNet: a single object in the center of the frame that belongs to one of the [image classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). We'll use one of the famous [monkey selfies](https://en.wikipedia.org/wiki/Monkey_selfie_copyright_dispute), and resize it appropriately:

```python
img = po.data.macaque()
# here we downsample the original image by a factor of 4 and then lop off the bottom.
# that way, when we take the central 224 pixels in the following block, we end up with a
# decent image.
img = po.process.blur_downsample(img, 2)[..., :-59, :]
img = po.process.center_crop(img, transform.crop_size[0])
po.plot.imshow(img, as_rgb=True);
```

###  Last steps
Now we create our model by passing the neural network, target layer, and preprocessing transform to plenoptic's {class}`~plenoptic.models.DeepNetFeatures`

```python
model = po.models.DeepNetFeatures(deepnet, target_layer, norm)
```

Finally, let's remove the gradient from all model parameters (as models in plenoptic [are fixed](remove-grad-doc)), convert everything to float64, for [reproducibility](float64-doc), and move everything to `DEVICE`:

```python
img = img.to(DEVICE).to(torch.float64)
model.to(DEVICE).to(torch.float64)
deepnet.to(DEVICE).to(torch.float64)
po.remove_grad(model)
```

## Visualizing classification of the clean image
First let us extract all the ImageNet categories

```python
imagenet_categories = np.asarray(weights.meta['categories'])
```

Let us define two helper functions. `convert_logits_to_probs` converts logits to probabilities that sum to 1. For an input image, `get_category` returns the probability vector (of length 1000) and the category with the highest probability.

```python
def convert_logits_to_probs(logits):
    return torch.nn.functional.softmax(logits, dim=1).squeeze()

def get_category(image):
    category_probs = convert_logits_to_probs(deepnet(norm(image))).detach().cpu()
    category = imagenet_categories[category_probs.argmax()]
    return category_probs, category
```

ResNet50 is trained to classify images into one of [1000 categories](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). The category, [guenon](https://en.wikipedia.org/wiki/Guenon), is an Old World monkey. Though it isn't the actual species of the monkey in question (a [Celebes crested macaque](https://en.wikipedia.org/wiki/Celebes_crested_macaque)), it's a reasonable category for it. Notice the model is also highly confident (probability of ~0.8) in its prediction.

```python
category_probs, category  = get_category(img)
po.plot.stem_plot(category_probs, title=category);
```

## Define optimized and reference metric

To qualify as an adversarial example, the image must satisfy two requirements: (1) the perturbation in image space is small and (2) the model outputting an incorrect classification with high confidence (cite papers). Conveniently, we already have these two ingredients built into the MAD competition framework and it is through carefully defining the reference and optimized metrics. More concretely, if we define the reference metric in pixel space and ask it to not change or minimally change, and define the optimized metric in representation space, which we want to make the synthesized image representation as different from the original, we would be able to meet the two requirements. For the reference metric, we use the the simple Mean Squared Error (MSE). A small MSE value between two images means the pixel values are not very different from each other.

```python
reference_metric = lambda x,y: po.metric.mse(x,y).mean() #.mean() averages across the RGB channels
```

For the optimized metric, we use the MSE of output logits in the last layer of the network.

```python
logit_mse = lambda x, y: po.metric.mse(model(x),model(y))
```

To encourage "one-hot" behaviour (model being highly confident in the class it chooses), we add an additional "penalty" term to the metric. It computes the sum of MAD image category probabilities raised to the power of 10. Note that in MAD competition, for a metric (x, y), x is the original image, y is the MAD image.

```python
exponent = 10
penalty_y = lambda y: torch.pow(convert_logits_to_probs(model(y)), exponent).sum()
```

To illustrate why this penalty term will encourage one-hot behaviour, we consider two probability vectors, one one-hot vector and another with random probabilities. We see the one-hot vector has higher value than the random vector.

```python
one_hot_vec = torch.zeros(1000)
one_hot_vec[200] = 1 # set the 200th element to 1 to create a one-hot vector
print(f"The penalty value for the one hot vector is {torch.pow(one_hot_vec, exponent).sum()}")
random_vec = torch.rand(1000)
random_vec = random_vec/random_vec.sum() # normalizing probabilities
print(f"The penalty value for the random vector is {torch.pow(random_vec, exponent).sum()}")
```

We can scale the strength of the penalty through `penalty_one_hot_lambda`. In Plenoptic, a metric needs to satisfy the requirement of returning 0 for two identical inputs. To meet this requirement, we add `penalty_x`, which is the penalty calculated on the original image category probabilities, a fixed number in the optimization. Putting everything together, we finally have the optimized metric.

```python
penalty_one_hot_lambda = 0.1
penalty_x = lambda x: torch.pow(convert_logits_to_probs(model(x)), exponent).sum()
optimized_metric = lambda x,y: logit_mse(x,y) + penalty_one_hot_lambda*(penalty_y(y) - penalty_x(x))
```

## Synthesize the adversarial image

We want to maximize the {attr}`~plenoptic.MADCompetition.optimized_metric` value while holding {attr}`~plenoptic.MADCompetition.reference_metric` fixed. {attr}`~plenoptic.MADCompetition.metric_tradeoff_lambda` controls the relative weight of optimized metric loss and reference metric loss in the objective function. We have found a {attr}`~plenoptic.MADCompetition.metric_tradeoff_lambda` value of 1e10 tends to produce good results for the purpose of this exercise.

```python
mad = po.MADCompetition(img, optimized_metric, reference_metric, "max", metric_tradeoff_lambda=1e10)
```

We set the initial noise to be a small value so the initial image is closer to the solution that we want: an image with close pixel values as the original but produces large changes in the representation

```python
mad.setup(initial_noise=0.001)
```

Running the synthesis generates an image that looks just like the original image but we see the optimized metric loss has increased significantly. Even though the reference metric loss measured in pixel space has also increased a little bit, it is not nearly as big as the change in the representation space.

```python
mad.synthesize(1000)
po.plot.synthesis_status(mad);
```

## Visualizing the adversarial image

Let us compare how the synthesized image compares to the original and initial images. In the bottow row the original image is subtracted to visualize the changes in pixels. There is nothing in the left image. In the middle image we see barely visible noise. And in the last image we see a low amount of pixel noise.

```python
imgs = [img, mad.initial_image, mad.mad_image]
mse = [po.metric.mse(img, i) for i in imgs]
titles = ['Original image', 'Initial image', 'Adversarial image']
diffs = [(i+1)/2 for i in [img-img, mad.initial_image-img, mad.mad_image-img]]
titles.extend([f"MSE={m.mean().item():.2e}" for m in mse])
imgs.extend(diffs)
po.plot.imshow(imgs, as_rgb=True, title=titles, col_wrap=3);
```

We can also visualize the difference in each color channel for the initial image and synthesized image.

```python
channelwise_diffs = [mad.initial_image-img, mad.mad_image-img]
po.plot.imshow(channelwise_diffs, col_wrap=3);
```

Finally let us visualize the category probabilities of the original, initial, and advesarial images using stem plots. In addition to the most likely category, we also show any category that has probability higher than 0.05. We see that the network thinks the synthesized image contains a cheeseburger with almost 100% certainty!

```python
fig, axes = plt.subplots(3, 2, figsize=(12, 20))
for i, img in enumerate([mad.image, mad.initial_image, mad.mad_image]):
    category_probs, category = get_category(img)
    likely_cats = '\n- '.join(list(imagenet_categories[category_probs>.05]))
    most_likely_cat = imagenet_categories[category_probs.argmax()]
    po.plot.imshow(img, ax=axes[i, 0], as_rgb=True, title=most_likely_cat)
    po.plot.stem_plot(category_probs, ax=axes[i,1], ylim=False)
    axes[i,1].set_title("Categories")
    axes[i,0].xaxis.set_visible(False)
    axes[i,0].yaxis.set_visible(False)
    axes[i,1].text(1, .5, f"Likely categories:\n- {likely_cats}", transform=axes[i,1].transAxes)
```
