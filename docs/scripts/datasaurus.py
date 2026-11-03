#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

import plenoptic as po

metamer_tarball = po.data.fetch_data("datasaurus_metamers.tar.gz")


def single_scatter(xy, title=None, xlim=(0, 100), ylim=(0, 100), **scatter_kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), layout="compressed")
    scatter_kwargs.setdefault("s", 5)
    ax.scatter(*xy, **scatter_kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect(1)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(1))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoLocator())
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(1))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoLocator())
    ax.set(xticklabels=[], yticklabels=[], xlim=xlim, ylim=ylim)
    return ax


def plenoptic_logo():
    title = "plenoptic-logo"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def circle():
    title = "circle"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def bullseye():
    title = "bullseye"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def away():
    title = "away"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def dots():
    title = "dots"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def hlines():
    title = "hlines"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def vlines():
    title = "vlines"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def xshape():
    title = "xshape"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def slantup():
    title = "slantup"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def slantdown():
    title = "slantdown"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def polygons():
    title = "polygons"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def oval():
    title = "oval"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def hwidelines():
    title = "hwidelines"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def vwidelines():
    title = "vwidelines"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)


def star():
    title = "star"
    data = torch.load(metamer_tarball / f"datasaurus-{title}.pt")["_metamer"]
    single_scatter(data.detach(), title)
