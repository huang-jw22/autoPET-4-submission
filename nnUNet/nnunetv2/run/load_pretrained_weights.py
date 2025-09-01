import os
import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()))
    else:
        saved_model = torch.load(fname, weights_only=False)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
        'organ_seg_layers'
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    expanded_stem_keys = []
    remapped_stem_keys = []
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        # Replicate / remap the stem to fit the number of input channels
        if os.environ.get("STEM") == "autoPET":
            if 'encoder.stem' in key:
                # original pretrained checkpoints have per-dataset variants.
                # Insert dataset id (221) into the key used in the checkpoint.
                if 'decoder.encoder.stem' in key:
                    src_key = key[:21] + "221" + '.' + key[21:]
                else:
                    src_key = key[:13] + "221" + '.' + key[13:]
                if src_key in pretrained_dict:
                    pretrained_dict[key] = pretrained_dict[src_key]
                    if verbose:
                        print(f"[load_pretrained] Remapped stem param: {key} <- {src_key} (shape {pretrained_dict[key].shape})")
                    remapped_stem_keys.append(key)
                else:
                    if verbose:
                        print(f"[load_pretrained][WARN] Expected stem source key missing in pretrained weights: {src_key}")
        else:
            # if 'stem' in key:
            #     pretrained_dict[key] = pretrained_dict[key].repeat(1, 2, 1, 1, 1) if len(pretrained_dict[key].shape) == 5 else pretrained_dict[key]
            skip_strings_in_pretrained = [
            '.seg_layers.',
            '.stem'
            ]

        # Channel expansion 2 -> 4 (only weight tensors of stem)
        if 'encoder.stem' in key and key.endswith('weight') and key in pretrained_dict:
            w_pre = pretrained_dict[key]
            w_cur = model_dict[key]
            if w_pre.shape != w_cur.shape:
                # handle 3D (5D tensor) or 2D (4D tensor)
                if w_pre.shape[2:] == w_cur.shape[2:] and w_pre.shape[0] == w_cur.shape[0] and w_pre.shape[1] < w_cur.shape[1]:
                    if verbose:
                        print(f"[load_pretrained] Expanding stem weight {key}: {w_pre.shape} -> {w_cur.shape}")
                    new_w = w_cur.clone()
                    # copy existing input channels
                    new_w[:, :w_pre.shape[1]] = w_pre
                    # zero-init new channels so network starts identical w.r.t. old channels
                    new_w[:, w_pre.shape[1]:] = 0
                    pretrained_dict[key] = new_w
                    expanded_stem_keys.append(key)
                else:
                    if verbose:
                        print(f"[load_pretrained][WARN] Shape mismatch for {key} cannot be auto-expanded: "
                              f"pre {w_pre.shape} vs model {w_cur.shape}")

        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model " \
                f"does not seem to be compatible with your network."



    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)

    # print(f"[load_pretrained] Remapped stem params: {len(remapped_stem_keys)}; Expanded stem weights: {len(expanded_stem_keys)}")
    # if expanded_stem_keys:
    #     for k in expanded_stem_keys:
    #         print(f"[load_pretrained] Expanded: {k} final shape {pretrained_dict[k].shape}")


