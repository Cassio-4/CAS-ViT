from model import rcvit
import torch
import torch.nn as nn
import math
from utils import load_state_dict

class Adapter(nn.Module):
    def __init__(self,
                 n_embd = None,
                 down_size = None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 layer_type="linear"):
        super().__init__()
        if (n_embd is None) or (down_size is None):
            print("WARNING: Invalid adapter sizes")
            raise NotImplementedError
        self.n_embd = n_embd
        self.down_size = down_size
        self.layer_type=layer_type

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        if ("linear" in layer_type):
            self.down_proj = nn.Linear(self.n_embd, self.down_size)
            self.non_linear_func = nn.ReLU()
            self.up_proj = nn.Linear(self.down_size, self.n_embd)
        elif("conv" in layer_type):
            self.down_proj = nn.Conv2d(in_channels=self.down_size[1],out_channels=self.down_size[1]//2,kernel_size=n_embd["conv_k"],stride=n_embd["conv_s"])
            self.non_linear_func = nn.ReLU()
            self.pool = nn.MaxPool2d(n_embd["pool_k"], n_embd["pool_s"])
            self.up_proj = nn.ConvTranspose2d(in_channels=self.down_size[1]//2,out_channels=self.down_size[1],kernel_size=n_embd["convT_k"],stride=n_embd["convT_s"] )

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        if("conv" in self.layer_type):
            down = self.pool(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class RCViTAdapter(rcvit.RCViT):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=..., norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, num_classes=1000, 
                 drop_rate=0, drop_path_rate=0, fork_feat=False, init_cfg=None, pretrained=None, distillation=True, adapter_config=None, checkpoint_path=None,
                 adapter_type="linear", **kwargs):
        super().__init__(layers, embed_dims, mlp_ratios, downsamples, norm_layer, attn_bias, act_layer, num_classes, drop_rate, 
                         drop_path_rate, fork_feat, init_cfg, pretrained, distillation, **kwargs)
        
        self.adapter_config = adapter_config
        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()
        self.adapter_type = adapter_type
        self.get_new_adapter()
        self.head = torch.nn.Linear(220, num_classes)
        ## Load pretrained weights
        if pretrained and checkpoint_path is not None:
            print("loading weights")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model"]
            load_state_dict(self, state_dict)
        ## Freeze all but adapter layers     
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        print(f"before freeze params: {n_parameters}")
        self.freeze()

    def get_embedding_dimensions(self):
        lst_dims = []
        x = torch.rand((1, 3, 224, 224))
        with torch.no_grad():
            x = self.patch_embed(x)
            for idx, block in enumerate(self.network):
                x = block(x)
                lst_dims.append(x.shape)
        return lst_dims
    
    def get_conv_adapter_config(self, dims):
        channel_height = dims[2]
        if channel_height == 56:
            conv_config = {"conv_k":(3,3), "conv_s":1, "pool_k":(3,3), "pool_s":2, "convT_k":(6,6), "convT_s":2}
        elif channel_height == 28:
            conv_config = {"conv_k":(3,3), "conv_s":1, "pool_k":(3,3), "pool_s":2, "convT_k":(6,6), "convT_s":2}
        elif channel_height == 14:
            conv_config = {"conv_k":(2,2), "conv_s":1, "pool_k":(3,3), "pool_s":2, "convT_k":(4,4), "convT_s":2}
        elif channel_height == 7:
            conv_config = {"conv_k":(2,2), "conv_s":1, "pool_k":(2,2), "pool_s":2, "convT_k":(3,3), "convT_s":2}
        else:
            raise(ValueError("Invalid dims for adapter"))
        return conv_config

    def get_new_adapter(self):
        lst_dims = self.get_embedding_dimensions()
        if True: #TODO: flag if adapter or not
            for i in range(len(self.network)):
                if "linear" in self.adapter_type:
                    embd_in = lst_dims[i][2]
                    adapter = Adapter(n_embd=embd_in, down_size=embd_in//2, dropout=0.1,
                                            init_option=self.adapter_config.ffn_adapter_init_option,
                                            adapter_scalar=self.adapter_config.ffn_adapter_scalar,
                                            adapter_layernorm_option=self.adapter_config.ffn_adapter_layernorm_option,
                                            layer_type=self.adapter_type
                                            )
                elif "conv" in self.adapter_type:
                    conv_config = self.get_conv_adapter_config(lst_dims[i])
                    adapter = Adapter(n_embd=conv_config,down_size=lst_dims[i], dropout=0.1,
                                            init_option=self.adapter_config.ffn_adapter_init_option,
                                            adapter_scalar=self.adapter_config.ffn_adapter_scalar,
                                            adapter_layernorm_option=self.adapter_config.ffn_adapter_layernorm_option,
                                            layer_type=self.adapter_type
                                            )
                else:
                    raise(ValueError("Invalid layer type for adapter"))
                self.cur_adapter.append(adapter)

            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")
    
    def freeze(self):
        for name, param in self.named_parameters():
            if ("cur_adapter" not in name):
                param.requires_grad = False
            if "head" in name:
                param.requires_grad = True

        for adapter in self.cur_adapter:
            for param in adapter.parameters():
                param.requires_grad = True
    
    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            residual = x
            adapt = self.cur_adapter[idx]
            x = adapt(x)
            if self.adapter_config.ffn_adapt:
                if self.adapter_config.ffn_option == 'sequential':
                    pass
                elif self.adapter_config.ffn_option == 'parallel':
                    x = x + residual
                else:
                    raise ValueError(self.config.ffn_adapt)
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out
