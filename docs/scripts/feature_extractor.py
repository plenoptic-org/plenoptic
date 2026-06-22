#!/usr/bin/env python3

import argparse
import functools
import urllib

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import plenoptic as po

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72


def torchvision_setup():
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    deepnet = torchvision.models.resnet50(weights=weights)
    deepnet.eval()
    transform = weights.transforms()
    norm = torchvision.transforms.Normalize(transform.mean, transform.std)
    crop = functools.partial(po.process.center_crop, output_size=transform.crop_size[0])
    imagenet_categories = np.asarray(weights.meta["categories"])

    def get_category(image, thresh=0.1):
        image_cat = po.to_numpy(
            torch.nn.functional.softmax(deepnet(norm(image)), dim=1).squeeze()
        )
        return imagenet_categories[image_cat > thresh]

    return deepnet, norm, crop, get_category


def timm_setup():
    deepnet = timm.create_model("timm/resnet50.tv_in1k", pretrained=True)
    deepnet.eval()
    transform = create_transform(
        **resolve_data_config(deepnet.pretrained_cfg, model=deepnet)
    )
    norm = transform.transforms[-1]
    crop = transform.transforms[1]

    r = urllib.request.urlopen(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    imagenet_categories = np.asarray(r.read().decode().split("\n"))

    def get_category(image, thresh=0.1):
        image_cat = po.to_numpy(
            torch.nn.functional.softmax(deepnet(norm(image)), dim=1).squeeze()
        )
        return imagenet_categories[image_cat > thresh]

    return deepnet, norm, crop, get_category


def prepare_image(crop):
    img = po.data.macaque()
    # here we downsample the original image by a factor of 4 and then lop off the
    # bottom. that way, when we take the central 224 pixels in the following block, we
    # end up with a decent image.
    img = po.process.blur_downsample(img, 2)[..., :-59, :]
    img = crop(img)
    return img


def get_success_measures(met, get_category):
    original_cat = get_category(met.image)
    metamer_cat = get_category(met.metamer)
    stacked_images = torch.cat([met.model(met.metamer), met.model(met.image)], 0)
    pearson_r = torch.corrcoef(stacked_images)[0, 1].item()
    return original_cat, metamer_cat, pearson_r


def main(target_layer="layer3", model_zoo="torchvision"):
    if model_zoo == "torchvision":
        deepnet, norm, crop, get_category = torchvision_setup()
    elif model_zoo == "timm":
        deepnet, norm, crop, get_category = timm_setup()
    model = po.models.FeatureExtractorModel(deepnet, target_layer, norm)
    po.remove_grad(model)
    img = prepare_image(crop)
    get_category(img)
    met = po.Metamer(img, model)
    met.to(torch.float64)
    met.load(po.data.fetch_data(f"ResNet50-{target_layer}_macaque_metamer.pt"))
    fig = po.plot.synthesis_status(met, figsize=(15, 4.5))
    return fig


def get_stats(target_layer="layer3"):
    deepnet, norm, crop, get_category = torchvision_setup()
    model = po.models.FeatureExtractorModel(deepnet, target_layer, norm)
    po.remove_grad(model)
    img = prepare_image(crop)
    met = po.Metamer(img, model)
    met.to(torch.float64)
    met.load(po.data.fetch_data(f"ResNet50-{target_layer}_macaque_metamer.pt"))
    return get_success_measures(met, get_category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Load in some example ResNet50 metamers, using FeatureExtractorModel."
        ),
    )
    parser.add_argument("--target_layer", "-l", default="layer3")
    parser.add_argument("--model-zoo", "-m", default="torchvision")
    parser.add_argument(
        "--save_path", "-p", default=None, help="Path to save synthesis status figure."
    )
    args = vars(parser.parse_args())
    save_path = args.pop("save_path")
    fig = main(**args)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
