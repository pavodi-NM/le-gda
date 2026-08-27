"""ImageNet loader over this project's `imagenet_subset` tree.

NOT part of the upstream Tip-Adapter repo. Written for
the README, *Comparative methods under our protocol*: upstream's own
`datasets/imagenet.py` calls `torchvision.datasets.ImageNet`, which needs the
full ~155 GB release plus devkit, and `imagenet` is absent from
`dataset_list` entirely. This is a DATA-ACCESS ADAPTER, not a method change --
nothing here touches the cache, the adapter, the loss or the search.

**The split rule is copied exactly from `scripts/extract_imagenet_subset.py`**
so that this loader addresses the SAME images the cached tensors were built
from, which is what makes `protocol_adapter.assert_pool_matches` pass:

    train/<c>/*   -> all 32 per class, the draw pool  (sorted os.listdir)
    val/<c>/*     -> sorted, then random.Random(c).shuffle(range(50));
                     idx[:10] -> val, idx[10:] -> test

`num_shots <= 0` returns the FULL 32/class pool, which is what
`protocol_adapter._pool_items` asks for. A positive `num_shots` defers to
`DatasetBase.generate_fewshot_dataset`, but note that under our protocol the
support set is injected by `protocol_adapter.support_items` and this path is
never what the trained arm sees -- it exists only so the constructor behaves
sanely when `main.py` builds the dataset for its classnames and template.

DELIBERATELY NOT COPIED: upstream Tip-Adapter's ImageNet quirk of setting
`val = test =` the 50k val set, which searches hyperparameters on the test
split. Our 10/class val is disjoint from the 40/class test. P3.
"""

import os
import random

from .utils import Datum, DatasetBase
from .imagenet_classnames import imagenet_classes

# Tip-Adapter's own 7-template ImageNet bank (main_imagenet.py / imagenet.py).
imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

template = imagenet_templates


class ImageNetSubset(DatasetBase):

    dataset_dir = 'imagenet_subset'

    def __init__(self, root, num_shots):
        self.image_dir = os.path.join(root, self.dataset_dir)
        self.template = imagenet_templates

        val, test, train = [], [], []
        for c in range(len(imagenet_classes)):
            name = imagenet_classes[c]

            d = os.path.join(self.image_dir, 'val', str(c))
            paths = sorted(os.listdir(d))
            idx = list(range(len(paths)))
            random.Random(c).shuffle(idx)
            val += [Datum(impath=os.path.join(d, paths[i]), label=c,
                          classname=name) for i in idx[:10]]
            test += [Datum(impath=os.path.join(d, paths[i]), label=c,
                           classname=name) for i in idx[10:]]

            dt = os.path.join(self.image_dir, 'train', str(c))
            train += [Datum(impath=os.path.join(dt, p), label=c,
                            classname=name) for p in sorted(os.listdir(dt))]

        if num_shots is not None and num_shots > 0:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
