"""Class-name bank for `imagenet_subset.py`.

NOT part of the upstream repo. `datasets/imagenet.py` already carries
Tip-Adapter's own 1000-name list, in the canonical wnid-sorted order the
`imagenet_subset` tree is laid out in; this module only re-exports it so the
loader in `patches/imagenet_subset_for_tip_adapter.py` can be copied in
verbatim. No names are redefined here -- a second copy of the list could drift
from the one the cached tensors were built against.
"""

from .imagenet import imagenet_classes  # noqa: F401
