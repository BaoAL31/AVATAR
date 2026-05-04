import os

import torch


UNIGRAM1000_LIST = ['<blank>'] + [_.split()[0] for _ in open(os.path.join(os.path.dirname(__file__), "labels", "unigram1000_units.txt")).read().splitlines()] + ['<eos>']


# Writes list of objects (anything that can be converted to str) to a txt file, separated by "\n"s
def write_to_txt(obj_list, path):
    f = open(path, "w")
    for obj in obj_list:
        f.write(str(obj) + "\n")
    f.close()
    

def ids_to_str(token_ids, char_list):
    tokenid_as_list = list(map(int, token_ids))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    return "".join(token_as_list).replace("<space>", " ")


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
        

def average_checkpoints(last):
    avg = None
    valid_count = 0
    for path in last:
        ckpt = torch.load(path, map_location="cpu")
        if "state_dict" not in ckpt:
            print(f"Warning: Checkpoint {path} has no 'state_dict' key. Skipping.")
            continue
        states = ckpt["state_dict"]
        # Strip "model." prefix from keys (PyTorch Lightning wraps model in "model." namespace)
        filtered = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
        if not filtered:
            print(f"Warning: Checkpoint {path} has no keys starting with 'model.'. Skipping.")
            continue
        if avg is None:
            avg = filtered
            valid_count = 1
        else:
            # Check for key mismatch
            if set(avg.keys()) != set(filtered.keys()):
                missing = set(avg.keys()) - set(filtered.keys())
                extra = set(filtered.keys()) - set(avg.keys())
                print(f"Warning: Checkpoint {path} has key mismatch. Missing: {missing}, Extra: {extra}. Skipping.")
                continue
            for k in avg.keys():
                avg[k] += filtered[k]
            valid_count += 1

    if avg is None or valid_count == 0:
        print("Error: No valid checkpoints to average.")
        return None

    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= valid_count
            else:
                avg[k] //= valid_count
    
    return avg


def _encoder_block_index(name: str) -> int:
    """Block index N from ...encoders.N... (works with or without backbone. prefix)."""
    parts = name.split(".")
    i = parts.index("encoders")
    return int(parts[i + 1])


def get_param_groups(model, num_blocks, base_lr_enc, base_lr_other, lr_decay_rate, min_lr=1e-6):
    param_groups = {}
    layer_scales = list(lr_decay_rate ** (num_blocks - i - 1) for i in range(num_blocks))
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "backbone.encoder.after_norm" in name:
            group_name = "after_norm"
            base_lr = max(base_lr_enc, min_lr)
        elif "backbone.encoder.embed" in name:
            group_name = "embed"
            base_lr = max(layer_scales[0] * base_lr_enc, min_lr)
        elif "backbone.encoder.frontend" in name or "backbone.encoder.linear" in name:
            group_name = "frontend"
            base_lr = max(layer_scales[0] * base_lr_enc, min_lr)
        elif "backbone.encoder.encoders" in name:
            group_id = _encoder_block_index(name)
            group_name = f"block_{group_id}"
            base_lr = max(layer_scales[group_id] * base_lr_enc, min_lr)
        elif "backbone.encoder.au_fusion" in name:
            group_name = "au_fusion"
            base_lr = max(base_lr_other, min_lr)
        else:
            # Keep unknown encoder submodules trainable under "other" instead of crashing.
            if name.startswith("backbone.encoder"):
                group_name = "other"
                base_lr = max(base_lr_other, min_lr)
            else:
                group_name = "other"
                base_lr = max(base_lr_other, min_lr)
            if name.startswith("target_backbone"):
                print(name)
        
        if group_name not in param_groups:
            param_groups[group_name] = {
                "name": group_name,
                "lr": base_lr,
                "params": []
            }
        param_groups[group_name]["params"].append(param)
    
    return list(param_groups.values())


def get_param_groups_ft(model, num_blocks, base_lr_enc, base_lr_other, lr_decay_rate, min_lr=1e-6):
    param_groups = {}
    layer_scales = list(lr_decay_rate ** (num_blocks - i - 1) for i in range(num_blocks))
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "encoder.after_norm" in name:
            group_name = "after_norm"
            base_lr = max(base_lr_enc, min_lr)
        elif "encoder.embed" in name:
            group_name = "embed"
            base_lr = max(layer_scales[0] * base_lr_enc, min_lr)
        elif "encoder.au_fusion" in name:
            group_name = "au_fusion"
            base_lr = max(base_lr_other, min_lr)
        elif "encoder.frontend" in name or "encoder.linear" in name:
            group_name = "frontend"
            base_lr = max(layer_scales[0] * base_lr_enc, min_lr)
        elif "encoder.encoders" in name:
            group_id = _encoder_block_index(name)
            group_name = f"block_{group_id}"
            base_lr = max(layer_scales[group_id] * base_lr_enc, min_lr)
        else:
            assert not name.startswith("encoder")
            group_name = "other"
            base_lr = max(base_lr_other, min_lr)
        
        if group_name not in param_groups:
            param_groups[group_name] = {
                "name": group_name,
                "lr": base_lr,
                "params": []
            }
        param_groups[group_name]["params"].append(param)
    
    return list(param_groups.values())
