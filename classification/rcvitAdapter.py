from model import rcvit
import torch
import torch.nn as nn
import copy
from vitEase import Adapter

class RCViTAdapter(rcvit.RCViT):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=..., norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, num_classes=1000, 
                 drop_rate=0, drop_path_rate=0, fork_feat=False, init_cfg=None, pretrained=None, distillation=True, **kwargs):
        super().__init__(layers, embed_dims, mlp_ratios, downsamples, norm_layer, attn_bias, act_layer, num_classes, drop_rate, 
                         drop_path_rate, fork_feat, init_cfg, pretrained, distillation, **kwargs)
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
            print(x.shape[2])
        return lst_dims

    def get_new_adapter(self):
        lst_dims = self.get_embedding_dimensions()
        if True: #TODO: flag if adapter or not
            for i in range(len(self.blocks)):
                adapter = Adapter(n_embd=1, down_size=1, dropout=0.1, bottleneck=config.ffn_num,
                                        init_option=config.ffn_adapter_init_option,
                                        adapter_scalar=config.ffn_adapter_scalar,
                                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                        ).to(self._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x, self.cur_adapter[idx])
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_test(self, x, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)
        
        features = []
        
        if use_init_ptm:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            features.append(x)
        
        for i in range(len(self.adapter_list)):
            x = copy.deepcopy(x_init)
            for j in range(len(self.blocks)):
                adapt = self.adapter_list[i][j]
                x = self.blocks[j](x, adapt)
            x = self.norm(x)
            features.append(x)
        
        x = copy.deepcopy(x_init)
        for i in range(len(self.blocks)):
            adapt = self.cur_adapter[i]
            x = self.blocks[i](x, adapt)
        x = self.norm(x)
        features.append(x)
        
        return features
