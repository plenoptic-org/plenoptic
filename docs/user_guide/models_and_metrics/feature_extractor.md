---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: plenoptic
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input]

import pooch

# don't have pooch output messages about downloading or untarring
logger = pooch.get_logger()
logger.setLevel("WARNING")
```

:::{admonition} Run this notebook yourself!
:class: important

Download the script: [`feature_extractor.py`](../../scripts/feature_extractor.py)!

:::

(feature_extractor)=
# Using Deep Neural Networks with plenoptic

:::{warning}
This notebook requires the optional dependency `torchvision`, which can be installed with `pip`.
:::

plenoptic is compatible with any model written in pytorch, including deep neural networks from the model zoos {external+torchvision:ref}`TorchVision <models>` and {external+timm:doc}`timm <models>`. In this notebook, we'll show how to adapt a deep net from these two packages for use with plenoptic, recreating some ResNet50 metamers shown in {cite:alp}`Feather2023-model-metam`, figure 2e.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import torch

import plenoptic as po

# this notebook uses torchvision, which is an optional dependency.
# if this fails, install torchvision in your plenoptic environment
# and restart the notebook kernel.
try:
    import torchvision
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "optional dependency torchvision not found!"
        " please install it in your plenoptic environment "
        "and restart the notebook kernel"
    )


dtype = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

%load_ext autoreload

%autoreload 2

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

# set seed for reproducibility
po.set_seed(0)
```

:::{admonition} This notebook retrieves cached synthesis results
:class: warning dropdown

The example metamer shown in this notebook takes about 15 minutes to synthesize on a GPU. Thus, instead of performing synthesis in this notebook, we have cached the result of it online and only download them for investigation.

:::


## Initializing the model

When synthesizing images for deep nets, as in {cite:alp}`Feather2023-model-metam`, it is common to pick a specific intermediate layer whose representation we wish to use. `torchvision` contains a "feature extractor" to grab activity from intermediate layers, and plenoptic's {class}`plenoptic.models.FeatureExtractorModel` is a small wrapper to simplify this process.

::::{tab-set}
:::{tab-item} layer2
:sync: layer2

```python
target_layer = "layer2"
```
:::
:::{tab-item} layer3
:sync: layer3

```python
target_layer = "layer3"
```
:::
:::{tab-item} layer4
:sync: layer4

```python
target_layer = "layer4"
```
:::
::::

In the rest of this section, we show how to initialize a plenoptic-compatible model using the weights from either the {external+torchvision:ref}`TorchVision <models>` or {external+timm:doc}`timm <models>` model zoos; their behavior after this section is the same.

First, we download the model weights for ResNet50 trained on [ImageNet-1K](https://en.wikipedia.org/wiki/ImageNet#ImageNet-1K) and initialize the `torchvision` / `timm` model.

::::{tab-set}
:::{tab-item} torchvision
:sync: torchvision

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 22-23
```

:::

:::{tab-item} timm
:sync: timm

Note that to run this cell (and the following `timm` cells), you must install `timm` as well (`pip install timm`)!

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 40
```
:::
::::

Next, we ensure that our model is in evaluation mode. Many models, including ResNet50, behave differently when in training and evaluation mode. In plenoptic, models are fixed and so we want the evaluation behavior:

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 24
```

Next, we need to specify the layer to target. If we look at the ResNet50 metamers in Figure 2e from {cite:alp}`Feather2023-model-metam`, we can see an interesting progression in layers 2 through 4: the layer 2 metamer looks almost identical to the target image, the layer 3 metamer starts to add RGB noise, and the layer 4 is almost completely unidentifiable, looking almost completely like random RGB noise.

Let's start with `"layer3"`, but note the metamer synthesis procedure in this notebook works with any of `"layer2"`, `"layer3"`, or `"layer4"` (and possibly others, they just haven't been tested).

:::{admonition} How do I know what layers I can use?
:class: dropdown question

You can view possible layer names with {external+torchvision:func}`torchvision.models.feature_extraction.get_graph_node_names`. (For more details on the node naming conventions, please see the {external+torchvision:ref}`About Node Names <about-node-names>` heading in the {external+torchvision:doc}`torchvision documentation <feature_extraction>`.)

```python
from torchvision.models import feature_extraction
# this function returns two lists, the first for training mode, the second for eval mode
feature_extraction.get_graph_node_names(deepnet)[1]
```

And note that you can specify multiple layers!
:::

Next, we grab the preprocessing transform from the model. As the [torchvision docs](https://docs.pytorch.org/vision/stable/models.html#using-the-pre-trained-models) explain it (quoting version `0.27`):

> Before using the pre-trained models, one must preprocess the image (resize with right resolution/interpolation, apply inference transforms, rescale the values etc). There is no standard way to do this as it depends on how a given model was trained. It can vary across model families, variants or even weight versions. Using the correct preprocessing method is critical and failing to do so may lead to decreased accuracy or incorrect outputs.

For models trained on ImageNet, this preprocessing consists of two steps: resizing to a height and width of 224 pixels and normalizing the color channels (subtracting means and dividing by standard deviations). Following {cite:alp}`Feather2023-model-metam`, we recommend including the normalization step in the model for metamer synthesis, but handling the image resizing externally. We demonstrate how to do so below.

Let's grab the normalizing transform and then initialize our plenoptic model:

::::{tab-set}
:::{tab-item} torchvision
:sync: torchvision

In torchvision, the transform is a single torch Module which we cannot easily subdivide, so we create a separate normalization transform, which we pass to {class}`~plenoptic.models.FeatureExtractorModel`:

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 25-26
```

```python
print(transform)
```

```python
ImageClassification(
    crop_size=[224]
    resize_size=[256]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BILINEAR
)
```

:::

:::{tab-item} timm
:sync: timm

In timm, the transform can be indexed into, so we can explicitly grab the normalization:

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 42-45
```

```python
print(transform)
```

```python
Compose(
    Resize(size=256, interpolation=bilinear, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    MaybeToTensor()
    Normalize(mean=tensor([0.4850, ...]), std=tensor([0.2290, ...]))
)
```

:::
::::

Finally, we'll pass our neural network, target layer, and preprocessing transform to plenoptic's {class}`~plenoptic.models.FeatureExtractorModel`:

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 85
```

Now, let's prepare the image. The input image needs to be an RGB image with a height and width of 224 pixels. It should probably also be like those found in ImageNet: a single object in the center of the frame that belongs to one of the [image classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). We'll use one of the famous [monkey selfies](https://en.wikipedia.org/wiki/Monkey_selfie_copyright_dispute), and resize it appropriately:

```{code-cell} ipython3
img = po.data.macaque()
# here we downsample the original image by a factor of 4 and then lop off the bottom.
# that way, when we take the central 224 pixels in the following block, we end up with a
# decent image.
img = po.process.blur_downsample(img, 2)[..., :-59, :]
```

How we crop the image down to 224 depends on which model zoo we're using:

::::{tab-set}
:::{tab-item} torchvision
:sync: torchvision

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 27
```

:::

:::{tab-item} timm
:sync: timm

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 46
```
:::
::::

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 68
```

```{code-cell} ipython3
:tags: [remove-cell]

target_layer = "layer2"
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
deepnet = torchvision.models.resnet50(weights=weights).eval()
transform = weights.transforms()
norm = torchvision.transforms.Normalize(transform.mean, transform.std)
model = po.models.FeatureExtractorModel(deepnet, target_layer, norm)
img = po.process.center_crop(img, transform.crop_size[0])
```

Let's visualize our resulting image:

```{code-cell} ipython3
po.plot.imshow(img, as_rgb=True);
```

ResNet50 is trained to classify images into one of [1000 categories](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). Any metamer of an intermediate layer should preserve this classification, which is the output of the final layer; this is one of the criteria that {cite:alp}`Feather2023-model-metam` check for synthesis success. Let's examine that classification now, creating a little helper function:

::::{tab-set}
:::{tab-item} torchvision
:sync: torchvision

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 28
```

:::

:::{tab-item} timm
:sync: timm

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 5,48-51
```
:::
::::

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 30-34,88
```

```python
guenon
```

The category, [guenon](https://en.wikipedia.org/wiki/Guenon), is an Old World monkey. Though it isn't the actual species of the monkey in question (a [Celebes crested macaque](https://en.wikipedia.org/wiki/Celebes_crested_macaque)), it's a reasonable category for it.

Finally, let's remove the gradient from all model parameters (as models in plenoptic [are fixed](remove-grad-doc)) and convert everything to float64, for [reproducibility](float64-doc):

```{code-cell} ipython3
po.remove_grad(model)
model.to(torch.float64)
img = img.to(torch.float64)
```

## Understanding the model

Our `model` <!-- skip-lint --> object now returns only the activations from our specified layer(s) as a single 2d vector (with the first dimension corresponding to the batch dimension of our input):

```{code-cell} ipython3
rep = model(img)
print(rep)
print(rep.shape)
```

We have flattened the model representation of the given layer (to support representations from multiple layers simultaneously). If you would like to retrieve the original shape, you can use the {func}`~plenoptic.models.FeatureExtractorModel.convert_to_dict` method:

```{code-cell} ipython3
rep = model.convert_to_dict(rep)
print(rep.keys())
print(rep[target_layer].shape)
```

{class}`~plenoptic.models.FeatureExtractorModel` also has a {func}`~plenoptic.models.FeatureExtractorModel.plot_representation` method, which creates two subplots. The first plots the average across channel, the average spatial representation, while the second averages across space to get a per-channel average representation:

```{code-cell} ipython3
fig, _ = model.plot_representation(rep)
```

## Synthesizing the metamer

:::{warning}
We do not perform synthesis in the exact same way as {cite:alp}`Feather2023-model-metam`. However, the resulting metamer is qualitatively similar. We note the differences below.
:::

Let us initialize our metamer object using the above image and model. Unlike in {cite:alp}`Feather2023-model-metam`, we are using the mean-squared error (the default for {class}`~plenoptic.Metamer`) as our loss function. We also initialize with a sample of uniformly-distributed noise whose values range from 0 to 1, whereas the paper initialized with "a sample from a normal distribution with a standard deviation of 0.05 and a mean of 0.5". Like that paper, we find better synthesis results if we use a learning-rate scheduler to halve the optimizer's learning rate regularly, using {class}`~torch.optim.lr_scheduler.StepLR` (see the following dropdown for more details):

```{code-cell} ipython3
met = po.Metamer(img, model)
met.to(DEVICE)
met.load(
    po.data.fetch_data(f"ResNet50-{target_layer}_macaque_metamer.pt"),
    map_location=DEVICE,
    tensor_equality_atol=1e-6,
)
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

These hyperparameters are the ones that work best for this target image. They should make a good starting point for other images, but you are encouraged to play around with the learning rate and scheduler!

Note that, as shown in the following block, `"layer2"` and `"layer3"` metamers were synthesized using the same hyperparameters, but we found better results for `"layer4"` with a slightly higher learning rate and slightly longer gaps before reducing learning rate size.

<!-- TestFeatureExtractor.test_resnet_macaque_metamer -->
```python
scheduler = torch.optim.lr_scheduler.StepLR
scheduler_kwargs = {
    "step_size": 5000 if target_layer == "layer4" else 3000,
    "gamma": 0.5
}
lr = 3e-2 if target_layer == "layer4" else 1e-2
met.setup(
    optimizer_kwargs={"lr": lr, "amsgrad": False},
    scheduler=scheduler,
    scheduler_kwargs=scheduler_kwargs
)
# by setting stop_iters_to_check=max_iter, we ensure it keeps going through
# all 12k iterations
met.synthesize(max_iter=12000, stop_iters_to_check=12000)
```
:::

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 96
```

:::{attention}

Depending upon how zoomed in your browser is, there may be some aliasing artifacts in the appearance of the metamers. If you see faint grid lines, you are encouraged to click on the `png` button to view the figure in its own tab and zoom in to avoid aliasing.

:::

::::{tab-set}
:::{tab-item} layer2
:sync: layer2

```{eval-rst}
.. plot:: scripts/feature_extractor_mpl.py layer2
  :include-source: false
```
:::
:::{tab-item} layer3
:sync: layer3

```{eval-rst}
.. plot:: scripts/feature_extractor_mpl.py layer3
  :include-source: false
```
:::
:::{tab-item} layer4
:sync: layer4

```{eval-rst}
.. plot:: scripts/feature_extractor_mpl.py layer4
  :include-source: false
```
:::
::::

In the above plots, we can see the metamer in the leftmost subplot, the loss over synthesis iterations in the middle, and the representation error on the right:
- Our metamers match the results discussed earlier in this notebook:  the layer 2 metamer looks almost identical to the target image, the layer 3 metamer starts to add RGB noise, and the layer 4 is almost completely unidentifiable, looking almost completely like random RGB noise.
- We can see that the optimization performed reasonably well: the loss decreased gradually over synthesis. If you were using these stimuli in an experiment (especially for `"layer4"`), it may be worth continuing a bit more to get the loss even lower, but these demonstrate the point.
- The representation error plot has the same structure as the {func}`~plenoptic.models.FeatureExtractorModel.plot_representation` plot above. We see that the error is fairly uniform across both space and channels.

The authors of {cite:alp}`Feather2023-model-metam` used two additional checks to verify that metamer synthesis had succeeded (quotes from "Results > Metamer optimization" section, pdf page 5):
- "the metamer had to result in the same classification decision by the model as the reference stimulus" (here, `guenon`):
- "measures of the match between the activations for the natural reference stimulus and its model metamer at the matched stage had to be much higher than would be expected by chance, as quantified with a null distribution". The authors used three measures here: Pearson and Spearman correlations and signal-to-noise ratio. Here, we show the Pearson correlation:

These can be computed as follows:

```{literalinclude} ../../scripts/feature_extractor.py
:dedent:
:lines: 73-76
```

And the following shows the result of this for each of our layers:

::::{tab-set}
:::{tab-item} layer2
:sync: layer2

```{eval-rst}
.. plot:: scripts/feature_extractor_mpl.py layer2_stats
  :include-source: false
```
:::
:::{tab-item} layer3
:sync: layer3

```{eval-rst}
.. plot:: scripts/feature_extractor_mpl.py layer3_stats
  :include-source: false
```
:::
:::{tab-item} layer4
:sync: layer4

```{eval-rst}
.. plot:: scripts/feature_extractor_mpl.py layer4_stats
  :include-source: false
```
:::
::::

We don't have the null distribution of correlations for this model. In order to truly verify synthesis success, one should compute these for each of the measures described above and verify the values for each the metamer.

In this notebook, we have demonstrated how to use deep neural networks from external models zoos with  {class}`plenoptic.models.FeatureExtractorModel`, and shown how to generate metamers for several intermediate layers.
