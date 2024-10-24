from timm.models import create_model
import torch
from torchvision import transforms
import utils as utils
import numpy as np
from PIL import Image
from model import *

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


model_name = "rcvit_xs"
model_weights = "/home/cassio/git/CAS-ViT/cas-vit-xs.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model: rcvit = rcvit_xs()
checkpoint = torch.load(model_weights, map_location="cpu", weights_only=False)
state_dict = checkpoint["model"]
utils.load_state_dict(model, state_dict)
model.to(device)

print(model.parameters)
model.eval()

with torch.inference_mode():
    #x = torch.rand((1, 3, 224, 224))
    x = load_image()
    out = model(x)
    print(out.argmax())
    # 910 Wooden Spoon
