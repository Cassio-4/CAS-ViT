from model import rcvit
import torch
import torch.nn as nn
import copy
from vitEase import Adapter

class RCViTAdapter(rcvit.RCViT):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=..., norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, num_classes=1000, 
                 drop_rate=0, drop_path_rate=0, fork_feat=False, init_cfg=None, pretrained=None, distillation=True, adapter_config=None, **kwargs):
        super().__init__(layers, embed_dims, mlp_ratios, downsamples, norm_layer, attn_bias, act_layer, num_classes, drop_rate, 
                         drop_path_rate, fork_feat, init_cfg, pretrained, distillation, **kwargs)
        self.adapter_config = adapter_config
        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()
    
    def get_embedding_dimensions(self):
        lst_dims = []
        x = torch.rand((1, 3, 224, 224))
        x = self.patch_embed(x)
        for idx, block in enumerate(self.network):
            x = block(x)
            lst_dims.append(x.shape)
        return lst_dims

    def get_new_adapter(self):
        lst_dims = self.get_embedding_dimensions()
        if True: #TODO: flag if adapter or not
            for i in range(len(self.network)):
                embd_in = lst_dims[i][2]
                adapter = Adapter(n_embd=embd_in, down_size=embd_in//2, dropout=0.1,
                                        init_option=self.adapter_config.ffn_adapter_init_option,
                                        adapter_scalar=self.adapter_config.ffn_adapter_scalar,
                                        adapter_layernorm_option=self.adapter_config.ffn_adapter_layernorm_option,
                                        )
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True
    
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
