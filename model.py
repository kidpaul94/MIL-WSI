import torch
import torch.nn as nn
import torchvision.models as models

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.2):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class EncoderBlock(nn.Module):
    '''Transformer encoder block.'''

    def __init__(
        self,
        num_heads: int = 6,
        emb_size: int = 768,
        expansion: int = 4,
        dropout: float = 0.2,
        attention_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = nn.LayerNorm(emb_size)
        self.self_attention = nn.MultiheadAttention(emb_size, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(emb_size)
        self.mlp = FeedForwardBlock(emb_size, expansion, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f'Expected (batch_size, seq_length, hidden_dim) got {input.shape}')
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 2, **kwargs):
        super().__init__(*[EncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 2):
        super(ClassificationHead, self).__init__()
        self.layernorm = nn.LayerNorm(emb_size)
        self.n_classes = nn.Linear(emb_size, n_classes)
        
    def forward(self, x):
        x = self.layernorm(x)
        x = x[:,0]
        x = self.n_classes(x)
        
        return x

class ViT_head(nn.Sequential):
    def __init__(self,
                 depth: int = 2,     
                 emb_size: int = 768,
                 n_classes: int = 2,
                 **kwargs):
        super().__init__(
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

class MIL_EFF(nn.Module):
    def __init__(self):
        super(MIL_EFF, self).__init__()
        path = 'EfficientNet_B4_Weights.DEFAULT'
        self.base_model = nn.Sequential(*(list(models.efficientnet_b4(weights=path).children())[:-2]))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=3)
        self.conv1x1 = nn.Conv2d(in_channels=1792, out_channels=768, kernel_size=1)
        self.cls_token = nn.Parameter(torch.randn(1,1, 768))
        self.head = ViT_head(depth=2, emb_size=768)
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.maxpool(x)
        x = self.conv1x1(x)
        N, C, H, W = x.size()
        x = x.view(1, C ,H*W*N)
        x = x.permute(0,2,1)
        x= torch.cat((self.cls_token, x), 1)
        x = self.head(x)
        
        return x
