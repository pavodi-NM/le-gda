import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *

# OUR PROTOCOL (clauses P1-P6; see scripts/protocol_adapter.py).
# Nothing below this import touches the model, the loss, AdamW, the cosine
# schedule, batch 256, augment_epoch, train_epoch or the augmentation pipeline.
from scripts.protocol_adapter import (support_items, eval_items,
                                      assert_pool_matches, report)


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    # PROTOCOL P6 / P1.
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--test-split', default='testfull')
    parser.add_argument('--backbone', default=None,
                        help="override cfg['backbone']; the configs ship RN50 only")
    # PROTOCOL knob, like --backbone: the configs pin shots at 16, and the
    # K in {1,2,4,8} sweep needs it on the command line. It sets how many
    # support images our seeded draw returns; it changes nothing about the
    # method, the cache, the loss or the search.
    parser.add_argument('--shots', type=int, default=None,
                        help="override cfg['shots']; the configs ship 16 only")
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))
    return acc, best_beta, best_alpha


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # PROTOCOL P3. Upstream evaluated on the TEST set here every epoch,
        # kept `best_acc`/`best_epoch` by test accuracy, checkpointed to
        # best_F_*.pt, and reloaded that checkpoint. All of it is deleted: no
        # test-set selection of any kind. The final epoch's adapter is used.
        adapter.eval()

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    clip_logits = 100. * test_features @ clip_weights
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    # PROTOCOL P3. Upstream printed max(best_acc, acc), i.e. the better of the
    # test-selected epoch and this val-searched evaluation. `best_acc` no
    # longer exists; this single val-searched number is the result.
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
    return acc, best_beta, best_alpha


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # Same dataset root as every other script (scripts/protocol_adapter.py),
    # so LEGDA_DATA_ROOT alone is enough; the configs' root_path is the default.
    cfg['root_path'] = os.environ.get('LEGDA_DATA_ROOT', cfg['root_path'])
    if args.backbone:
        cfg['backbone'] = args.backbone
    if args.shots:
        cfg['shots'] = args.shots

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # PROTOCOL P2. Reconstruction guard: the support pool we are about to draw
    # from must be byte-identical in label order to the cached pool every other
    # row of the table was fitted on. Fails loudly rather than silently
    # training on a different 16 images.
    tag = cfg['backbone'].replace('/', '-')
    assert_pool_matches(cfg['dataset'], tag)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # PROTOCOL P1. Our val and our uncapped standard test split, not
    # `dataset.test`. Both are fixed across seeds, so encode them once.
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    val_items = eval_items(cfg['dataset'], 'val')
    test_items = eval_items(cfg['dataset'], args.test_split)
    print(f"\nPROTOCOL: val={len(val_items)} test[{args.test_split}]={len(test_items)} "
          f"seeds={args.seeds}\n")
    val_loader = build_data_loader(data_source=val_items, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=test_items, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    base_cache_dir = cfg['cache_dir']
    # Eval features do NOT depend on the seed (is_train=False -> no
    # augmentation, deterministic transform), so they are encoded ONCE. This
    # changes no number; it removes a 3x re-encode of every eval split.
    os.makedirs(base_cache_dir, exist_ok=True)
    cfg['cache_dir'] = base_cache_dir
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    arms = {"TIP": [], "TIP-F": []}
    for seed in args.seeds:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        # Their RNG still drives loader shuffling and the augmentation stream;
        # it no longer drives the support draw. Seeded per our seed so the three
        # runs are genuinely distinct rather than sharing augmentation noise.
        random.seed(seed)
        torch.manual_seed(seed)

        cfg['cache_dir'] = os.path.join(base_cache_dir, f"seed{seed}")
        os.makedirs(cfg['cache_dir'], exist_ok=True)

        # PROTOCOL P2. OUR seeded K-shot draw, in place of upstream's
        # `dataset.train_x` (which is `generate_fewshot_dataset` over the full
        # train split). Same transform, same batch size, same shuffle flags.
        items, _, _ = support_items(cfg['dataset'], cfg['shots'], seed)
        train_loader_cache = build_data_loader(data_source=items, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
        train_loader_F = build_data_loader(data_source=items, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

        print("\nConstructing cache model by few-shot visual features and labels.")
        cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

        a_tip = run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)
        a_tipf = run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)
        arms["TIP"].append(a_tip[0])
        arms["TIP-F"].append(a_tipf[0])
        print(f"\n>>> seed {seed}: TIP={a_tip[0]:.2f} (beta={a_tip[1]:.3f} alpha={a_tip[2]:.3f}) | "
              f"TIP-F={a_tipf[0]:.2f} (beta={a_tipf[1]:.3f} alpha={a_tipf[2]:.3f})")

    print("\n" + "="*70)
    for k, v in arms.items():
        print(f"  {k:6s} per-seed {[round(x,2) for x in v]}  mean {sum(v)/len(v):.2f}")
    if args.out:
        report([{"backbone": cfg['backbone'], "dataset": cfg['dataset'],
                 "K": cfg['shots'], "test_split": args.test_split,
                 "epochs": cfg['train_epoch'], "impl": "upstream-main.py",
                 "arms": arms}], args.out,
               meta={"protocol": "P1-P6, scripts/protocol_adapter.py",
                     "seeds": args.seeds})


if __name__ == '__main__':
    main()