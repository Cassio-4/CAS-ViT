from timm.models import create_model
import torch
import torch.nn as nn
from torchvision import transforms
import utils as utils
import numpy as np
from PIL import Image
from model import *
from classification.model.rcvitAdapter import RCViTAdapter
from vitEase import VisionTransformer 
import argparse, json
from easydict import EasyDict


def get_args_parser():
    parser = argparse.ArgumentParser('CAS-ViT training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--patience_time', default=15, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='rcvit_xs', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--weights_path', type=str, default="/home/cassio/git/CAS-ViT/cas-vit-xs.pth",
                        help='Path to pretrained weights')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    # Data
    parser.add_argument('--data_path', default="/home/cassio/git/CAS-ViT/carros", type=str,
                        help='dataset path')
    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=6e-3, metavar='LR',
                        help='learning rate (default: 6e-3), with total batch size 4096')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
   
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--classifier_dropout', default=0.0, type=float)
    
    return parser.parse_args()

def load_image(path="/home/cassio/git/CAS-ViT/WoodenSpoon.jpeg"):
    image = Image.open(path)
    data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
    image = data_transforms(image).float()
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    print(image.shape)
    return image

def get_vision_config():
    par = argparse.ArgumentParser('discard', add_help=False)
    par.add_argument('--config', type=str, default="/home/cassio/git/CAS-ViT/classification/ease_cifar.json",
                        help='Json file of settings.')
    ar = par.parse_args()
    
    with open(ar.config) as data_file:
        param = json.load(data_file)
     
    ar = vars(ar)
    ar.update(param)
    tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=64,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = ar["device"][0]
            )
    return tuning_config

args = get_args_parser()
model_name = "rcvit_xs"
model_weights = "/home/cassio/git/CAS-ViT/cas-vit-xs.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

tuning_config=get_vision_config()

model = RCViTAdapter(layers=[2, 2, 4, 2], embed_dims=[48, 56, 112, 220], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, num_classes=1000, drop_rate=0., drop_path_rate=0.1,
        fork_feat=False, init_cfg=None,  pretrained=False, distillation=False, adapter_config=tuning_config)


#model1 = VisionTransformer(tuning_config=get_vision_config())
#print(model1.parameters)

checkpoint = torch.load(model_weights, map_location="cpu", weights_only=False)
state_dict = checkpoint["model"]
utils.load_state_dict(model, state_dict)
model.to(device) 
model.eval()
model.get_embedding_dimensions()


with torch.inference_mode():
    #x = torch.rand((1, 3, 224, 224))
    x = load_image()
    out = model(x)
    print(out.argmax())
    # 910 Wooden Spoon """
