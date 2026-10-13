"""ImageNet-val loaders.  Calibration and evaluation splits are disjoint.

The validation set must be in class-folder layout (1000 wnid folders); set
IMAGENET_VAL to its location if it is not at the default path.
"""
import os
import torch
from torchvision import datasets, transforms

IMAGENET = os.path.expanduser(os.environ.get("IMAGENET_VAL", "~/datasets/ILSVRC/val"))

_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _folder():
    ds = datasets.ImageFolder(IMAGENET, _TF)
    assert len(ds.classes) == 1000, f"expected 1000 wnid folders, got {len(ds.classes)}"
    return ds


def _per_class_index(ds, lo, hi):
    """Indices of the [lo, hi) -th image of every class (files are sorted)."""
    seen, out = {}, []
    for i, (_, y) in enumerate(ds.samples):
        k = seen.get(y, 0)
        if lo <= k < hi:
            out.append(i)
        seen[y] = k + 1
    return out


def calib_loader(n_per_class=1, batch_size=16, workers=8, seed=0, offset=0):
    """Calibration split: image 0 of each class, deterministically shuffled.

    The shuffle matters: callers consume only the first `calib` images, and the
    per-class index is grouped by class, so without it a 256-image calibration
    set would be drawn from the first 256 wnids alphabetically rather than from
    the whole label space.  `offset` skips that many images of the shuffled order,
    which gives a second calibration set disjoint from the first.
    """
    ds = _folder()
    idx = _per_class_index(ds, 0, n_per_class)
    perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))
    idx = [idx[i] for i in perm.tolist()][offset:]
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=True)


def heldout_loader(n_images, offset=0, batch_size=8, workers=8, seed=1):
    """Held-out images for the propagation diagnostics: image 10 of every class (the
    first image of the evaluation split, so disjoint from calibration), shuffled
    once with `seed`.  Returns images [offset, offset + n_images) of that order, so
    consecutive offsets give disjoint sets drawn from different classes.
    """
    ds = _folder()
    idx = _per_class_index(ds, 10, 11)
    perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))
    idx = [idx[i] for i in perm.tolist()][offset:offset + n_images]
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=True)


def eval_loader(n_per_class=10, batch_size=64, workers=8):
    """Evaluation split: images 10..10+n of each class -- disjoint from calibration."""
    ds = _folder()
    idx = _per_class_index(ds, 10, 10 + n_per_class)
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=True)


@torch.no_grad()
def top1(model, loader, device):
    model.eval()
    ok = tot = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ok += (model(x).argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * ok / tot
