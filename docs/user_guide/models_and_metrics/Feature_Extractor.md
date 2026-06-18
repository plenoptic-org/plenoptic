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

:::{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`Feature_Extractor.ipynb`**!

Run it in your browser: **{binder}`Feature_Extractor.ipynb`**!

:::

# Using Deep Neural Networks with plenoptic

:::{warning}
This notebook requires the optional dependency `torchvision`, which can be installed with `pip`.
:::

plenoptic is compatible with any model written in pytorch, including deep neural networks from the model zoos {external+torchvision:ref}`TorchVision <models>` and {external+timm:doc}`timm <models>`. In this notebook, we'll show how to adapt a deep net from these two packages for use with plenoptic, recreating some ResNet50 metamers shown in {cite:alp}`Feather2023-model-metam`, figure 2e.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
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

Additionally, while you can normally call {func}`~plenoptic.Metamer.synthesize` again to pick up where we left out, the cached version of the results shown here discarded the optimizer's state dict (to reduce the size on disk). Thus, calling `met.synthesize(100)` with one of our cached and loaded metamer objects **will not** give the same result as calling `met.synthesize(12100)` with a new metamer object initialized as shown in this notebook.

:::


## Initializing the model

When synthesizing images for deep nets, as in {cite:alp}`Feather2023-model-metam`, it is common to pick a specific intermediate layer whose representation we wish to use. `torchvision` contains a "feature extractor" to grab activity from intermediate layers, and plenoptic's {class}`plenoptic.models.FeatureExtractorModel` is a small wrapper to simplify this process.

```{code-cell} ipython3
target_layer = "layer3"
```

In the rest of this section, we show how to initialize a plenoptic-compatible model using the weights from either the {external+torchvision:ref}`TorchVision <models>` or {external+timm:doc}`timm <models>` model zoos; their behavior after this section is the same.

First, we download the model weights for ResNet50 trained on [ImageNet-1K](https://en.wikipedia.org/wiki/ImageNet#ImageNet-1K) and initialize the `torchvision` / `timm` model.

::::{tab-set}
:::{tab-item} torchvision
:sync: torchvision

```{code-block} python
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
tv_model = torchvision.models.resnet50(weights=weights)
```

:::

:::{tab-item} timm
:sync: timm

Note that to run this cell (and the following `timm` cells), you must install `timm` as well (`pip install timm`)!

```{code-block} python
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
timm_model = timm.create_model("timm/resnet50.tv_in1k", pretrained=True)
```
:::
::::

Next, we ensure that our model is in evaluation mode. Many models, including ResNet50, behave differently when in training and evaluation mode. In plenoptic, models are fixed and so we want the evaluation behavior:

::::{tab-set}
:::{tab-item} torchvision
:sync: torchvision

```{code-block} python
tv_model = tv_model.eval()
```

:::

:::{tab-item} timm
:sync: timm

```{code-block} python
timm_model = timm_model.eval()
```
:::
::::

Next, we need to specify the layer to target. If we look at the ResNet50 metamers in Figure 2e from {cite:alp}`Feather2023-model-metam`, we can see an interesting progression in layers 2 through 4: the layer 2 metamer looks almost identical to the target image, the layer 3 metamer starts to add RGB noise, and the layer 4 is almost completely unidentifiable, looking almost completely like random RGB noise.

Let's start with `"layer3"`, but note the metamer synthesis procedure in this notebook works with any of `"layer2"`, `"layer3"`, or `"layer4"` (and possibly others, they just haven't been tested).

:::{admonition} How do I know what layers I can use?
:class: dropdown question

You can view possible layer names with {external+torchvision:func}`torchvision.models.feature_extraction.get_graph_node_names`. (For more details on the node naming conventions, please see the {external+torchvision:ref}`About Node Names <about-node-names>` heading in the {external+torchvision:doc}`torchvision documentation <feature_extraction>`.)

```{code-block} python
from torchvision.models import feature_extraction
# this function returns two lists, the first for training mode, the second for eval mode
feature_extraction.get_graph_node_names(tv_model)[1]
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

```{code-block} python
tv_transform = weights.transforms()
print(tv_transform)
tv_norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
model = po.models.FeatureExtractorModel(tv_model, target_layer, tv_norm)
```

```{code-block} python
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

```{code-block} python
timm_transform = create_transform(
    **resolve_data_config(timm_model.pretrained_cfg, model=timm_model)
)
print(timm_transform)
timm_norm = timm_transform.transforms[-1]
model = po.models.FeatureExtractorModel(timm_model, target_layer, timm_norm)
```

```{code-block}
Compose(
    Resize(size=256, interpolation=bilinear, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    MaybeToTensor()
    Normalize(mean=tensor([0.4850, ...]), std=tensor([0.2290, ...]))
)
```

:::
::::

Now, let's prepare the image. The input image needs to be an RGB image with a height and width of 224 pixels. We'll use one of the famous [monkey selfies](https://en.wikipedia.org/wiki/Monkey_selfie_copyright_dispute), and resize it appropriately:

```{code-cell} ipython3
img = po.load_images(po.data.fetch_data("Macaca_nigra_self-portrait.jpg"), False)
# here we downsample the original image by a factor of 4 and then lop off the bottom.
# that way, when we take the central 224 pixels in the following block, we end up with a
# decent image.
img = po.process.blur_downsample(img, 2)[..., :-60, :]
```

::::{tab-set}
:::{tab-item} torchvision
:sync: torchvision

```{code-block} python
img = po.process.center_crop(img, tv_transform.crop_size[0])
```

:::

:::{tab-item} timm
:sync: timm

```{code-block} python
timm_crop = timm_transform.transforms[1]
img = timm_crop(img)
```
:::
::::

```{code-cell} ipython3
:tags: [remove-cell]

weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
tv_model = torchvision.models.resnet50(weights=weights).eval()
tv_transform = weights.transforms()
tv_norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
model = po.models.FeatureExtractorModel(tv_model, target_layer, tv_norm)
img = po.process.center_crop(img, tv_transform.crop_size[0])
```

Let's visualize our resulting image:

```{code-cell} ipython3
po.plot.imshow(img, as_rgb=True);
```

ResNet50 is trained to classify images into one of [1000 categories](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/). Any metamer of an intermediate layer should preserve this classification, which is the output of the final layer; this is one of the criteria that {cite:alp}`Feather2023-model-metam` check for synthesis success. Let's examine that classification now, creating a little helper function:

```{code-cell} ipython3
# adding this threshold means we'll get multiple categories if the model isn't sure, but
# for the examples in this notebook, there's only one good classification.
def get_category(image, thresh=0.1):
    imagenet_categories = np.asarray(weights.meta["categories"])
    image_cat = po.to_numpy(
        torch.nn.functional.softmax(tv_model(tv_norm(image)), dim=1).squeeze()
    )
    return imagenet_categories[image_cat > thresh]


get_category(img)
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
rep.shape
```

We have flattened the model representation of the given layer (to support representations from multiple layers simultaneously). If you would like to retrieve the original shape, you can use the {func}`~plenoptic.models.FeatureExtractorModel.convert_to_dict` method:

```{code-cell} ipython3
rep = model.convert_to_dict(rep)
print(rep.keys())
rep[target_layer].shape
```

{class}`~plenoptic.models.FeatureExtractorModel` also has a {func}`~plenoptic.models.FeatureExtractorModel.plot_representation` method, which creates two subplots. The first plots the average across channel, the average spatial representation, while the second averages across space to get a per-channel average representation:

```{code-cell} ipython3
model.plot_representation(rep)
```

## Synthesizing the metamer

:::{warning}
We do not perform synthesis in the exact same way as {cite:alp}`Feather2023-model-metam`. However, the resulting metamer is qualitatively similar. We will note the differences below.
:::

Let us initialize our metamer object using the above image and model. Unlike in {cite:alp}`Feather2023-model-metam`, we are using the mean-squared error (the default) as our loss function. Like that paper, we find better synthesis results if we use learning-rate scheduler to halve the optimizer's learning rate every 3000 iterations, using {class}`~torch.optim.lr_scheduler.StepLR` (the exception is for the `"layer4"` metamer, where we find better results without a scheduler):

```{code-cell} ipython3
met = po.Metamer(img, model)
met.load(po.data.fetch_data("ResNet50_macaque_metamer.pt"))
```

:::{admonition} How to run this synthesis manually
:class: dropdown note

<!-- TestFeatureExtractor.test_macaque_metamer -->
```{code-block} python
lr = 1e-2 if target_layer == "layer4" else 3e-3
scheduler = torch.optim.lr_scheduler.StepLR if target_layer != "layer4" else None
scheduler_kwargs = {"step_size": 3000, "gamma": 0.5}
met.setup(optimizer_kwargs={"lr": lr}, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs)
met.synthesize(max_iter=12000)
```
:::

```{code-cell} ipython3
po.plot.synthesis_status(met);
```

```{code-cell} ipython3
get_category(met.metamer)
```

```{code-cell} ipython3
pearson_corr = torch.corrcoef(torch.cat([model(met.metamer), model(met.image)], 0))[
    0, 1
].item()
```
