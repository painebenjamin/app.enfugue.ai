# type: ignore
# adapted from https://github.com/ProjectNUWA/DragNUWA
# -*- coding:utf-8 -*-
import os
import sys
import shutil
import yaml
import random
import importlib
from PIL import Image
from warnings import simplefilter
import imageio
import numpy as np
import torch
from torchvision import utils
from enfugue.util import logger

simplefilter(action='ignore', category=FutureWarning)

def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname

def data2file(data, filename, type=None, override=False, printable=False, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(filename) or override:
        if extname in ['jpg', 'png', 'jpeg']:
            utils.save_image(data, filename, **kwargs)
        elif extname == 'gif':
            imageio.mimsave(filename, data, format='GIF', duration=kwargs.get('duration'), loop=0)
        elif extname == 'txt':
            if kwargs is None:
                kwargs = {}
            max_step = kwargs.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w', encoding='utf-8') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('Do not support this type')
        if printable: logger.info('Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: logger.info(
            'Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))


def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    
    if extname in ['pth', 'ckpt']:
        data = torch.load(filename, map_location=kwargs.get('map_location'))
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            logger.info('Loaded data from %s' % os.path.abspath(filename))
    return data


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        logger.info('Removing dirname: %s' % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError('Failed to delete %s because %s' % (dirname, e))

    if not os.path.exists(dirname):
        logger.info('Making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def adaptively_load_state_dict(target, weights_file, device="cpu", dtype=None):
    from enfugue.diffusion.util import iterate_state_dict
    import torch

    target_dict = target.state_dict()

    unexpected_keys = []
    wrong_tensor_keys = []
    used_keys = []

    for k, v in iterate_state_dict(weights_file, device):
        if k in target_dict:
            if (
                isinstance(v, torch.Tensor) and
                isinstance(target_dict[k], torch.Tensor) and
                v.size() == target_dict[k].size()
            ):
                this_dtype = target_dict[k].dtype if dtype is None else dtype
                target_dict[k] = v.detach().clone().to(dtype=this_dtype)
                used_keys.append(k)
            else:
                wrong_tensor_keys.append(k)
        else:
            unexpected_keys.append(k)

    missing_keys = list(set(list(target_dict.keys()))-set(used_keys))
    target.load_state_dict(target_dict)
    del target_dict

    if device == "cuda":
        import torch.cuda
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps":        
        import torch.mps
        torch.mps.empty_cache()
        torch.mps.synchronize()

    if len(unexpected_keys) != 0:
        logger.warning(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(missing_keys) != 0:
        logger.warning(
            f"Some weights of state_dict are missing used in target {missing_keys}"
        )
    if len(wrong_tensor_keys) != 0:
        logger.warning(
            f"Some weights of state_dict are the wrong type or shape: {wrong_tensor_keys}"
        )
    if len(used_keys) == 0:
        logger.warning(
            "No weights were loaded from state_dict."
        )
        
    elif len(unexpected_keys) == 0 and len(missing_keys) == 0 and len(wrong_tensor_keys) == 0:
        logger.warning("Strictly loaded state_dict.")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def image2pil(filename):
    return Image.open(filename)


def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)


# 格式转换
def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr


def arr2pil(arr):
    if arr.ndim == 3:
        return Image.fromarray(arr.astype('uint8'), 'RGB')
    elif arr.ndim == 4:
        return [Image.fromarray(e.astype('uint8'), 'RGB') for e in list(arr)]
    else:
        raise ValueError('arr must has ndim of 3 or 4, but got %s' % arr.ndim)

def notebook_show(*images):
    from IPython.display import Image
    from IPython.display import display
    display(*[Image(e) for e in images])
