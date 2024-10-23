from timm.models import create_model
import argparse
import torch
from torchvision import transforms
import utils as utils
import numpy as np
from PIL import Image
from my_dataset import build_dataset
from model import *

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


args = get_args_parser()
device = "cuda" if torch.cuda.is_available() else "cpu"

model: rcvit = create_model(
        args.model_name,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        head_init_scale=1.0,
        input_res=args.input_size,
        classifier_dropout=args.classifier_dropout,
        distillation=False,
    )
checkpoint = torch.load(args.weights_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["model"]
utils.load_state_dict(model, state_dict)
#model.head = torch.nn.Linear(220, 2)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

dl_train, dl_valid, dl_test = build_dataset(args)
epochs = args.epochs
loss_train = []
loss_eval  = []

stop = False
epoch = 0
lowest_loss_eval = 10000
last_best_result = 0
while (not stop):
    model.train()
    lloss = []
    for x,y in dl_train:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        closs = criterion(pred,y)
        closs.backward()
        opt.step()
        opt.zero_grad()
        lloss.append(closs.item())
        #print(closs.item())
    loss_train.append(np.mean(lloss))
    lloss = []
    model.eval()
    lres = []
    ytrue = []
    with torch.no_grad():
        for data,y in dl_valid:
            data = data.to(device)

            pred = model(data)
            closs = criterion(pred.cpu(),y)
            lloss.append(closs.item())
            res  = pred.argmax(dim=1).cpu().tolist()
            lres += res
            ytrue += y
    avg_loss_eval = np.mean(lloss)
    loss_eval.append(avg_loss_eval)
    if avg_loss_eval < lowest_loss_eval:
        lowest_loss_eval = avg_loss_eval
        last_best_result = 0
        print("Best model found! saving...")
        actual_state = {'optim':opt.state_dict(),'model':model.state_dict(),'epoch':epoch,'loss_train':loss_train,'loss_eval':loss_eval}
        torch.save(actual_state,'best_model.pth')
    last_best_result += 1
    if last_best_result > args.patience_time:
        stop = True
    print("epoch %d loss_train %4.3f loss_eval %4.3f last_best %d"%(epoch,loss_train[-1],loss_eval[-1],last_best_result))
    epoch += 1

