"""Classnames + prompt templates per dataset, including imagenet.

The run scripts only ever use build_dataset for (classnames, template);
imagenet has no CoOp-style builder entry (and no split json), so it is
served from the hardcoded lists in datasets/imagenet.py instead.
"""

from datasets import build_dataset
from scripts.run_cache_retrieval_null import ROOT


def class_template(name):
    if name == "imagenet":
        from datasets.imagenet import imagenet_classes, imagenet_templates
        return imagenet_classes, imagenet_templates
    ds = build_dataset(name, ROOT, 1)
    return ds.classnames, ds.template
