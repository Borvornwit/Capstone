import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from x_transformers import Encoder

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

class LinearTrans(nn.Module):
    def __init__(self, in_channels: int = 3, emb_size: int = 768, img_height: int = 63, img_width: int = 300,kernel_height:int = 3, kernel_width: int = 30, stride_h:int = 1, stride_w:int=15):
        self.kernel_size = (kernel_height,kernel_width)
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Unfold(self.kernel_size,stride=(stride_h,stride_w)),
            #Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            Rearrange('b c s -> b s c'),
            nn.Linear(kernel_height * kernel_width * in_channels, emb_size)
        )


    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)
        x = self.projection(x)
        # print(x.shape)
        return x

class PositionEncoderRppg(nn.Module):
    def __init__(self, in_channels: int = 3, emb_size: int = 768, img_height: int = 63, img_width: int = 300,kernel_height:int = 3, kernel_width: int = 30, stride_h:int = 1, stride_w:int=15):
      super().__init__()
      self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
    #   print(((img_height-kernel_height+stride_h)//stride_h)*((img_width-kernel_width+stride_w)//stride_w) + 1)
      self.positions = nn.Parameter(torch.randn(((img_height-kernel_height+stride_h)//stride_h)*((img_width-kernel_width+stride_w)//stride_w) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class PositionEncoderFinalRppg(nn.Module):
    def __init__(self, in_channels: int = 3, emb_size: int = 768):
      super().__init__()
      self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 num_head: int = 8,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size,num_head, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHeadRppg(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

class ViTRppg(nn.Module):
    def __init__(self,
               in_channels: int = 3,
               kernel_height = 3,
               kernel_width = 30,
               emb_size: int = 768,
               img_height: int = 63,
               img_width: int = 300,
               stride_h: int = 1,
               stride_w: int = 15,
                depth: int = 6,
                n_classes: int = 2,
                **kwargs):
        super().__init__()
        self.linear = LinearTrans(in_channels,emb_size, img_height, img_width, kernel_height, kernel_width, stride_h, stride_w)
        self.pos = PositionEncoderRppg(in_channels,emb_size, img_height, img_width, kernel_height, kernel_width, stride_h, stride_w)
        # self.trans = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        self.trans = Encoder(dim=emb_size,depth = depth,heads = 3)
        self.classification = ClassificationHeadRppg(emb_size, n_classes)
        
    def forward(self,x):
      x = self.linear(x)
      x = self.pos(x)
      x = self.trans(x)
      x = self.classification(x[:,0,:])
      return x 

class TransRppg(nn.Module):
  def __init__(self,
               in_channels: int = 3,
               kernel_height = 3,
               kernel_width = 30,
               emb_size: int = 768,
               img_height: int = 63,
               img_width: int = 300,
               stride_h: int = 1,
               stride_w: int = 15,
                depth: int = 6,
                n_classes: int = 2,
                **kwargs):
        super().__init__()
        # self.patch_embedded_face = PatchEmbeddingTrans(in_channels,emb_size, img_height, img_width, kernel_height, kernel_width, stride_x, stride_y)
        # self.patch_embedded_bg = PatchEmbeddingTrans(in_channels,emb_size, 63, img_width, kernel_height, kernel_width, stride_x, stride_y)
        self.linear = LinearTrans(in_channels,emb_size, img_height, img_width, kernel_height, kernel_width, stride_h, stride_w)
        self.pos_face = PositionEncoderRppg(in_channels,emb_size, img_height, img_width, kernel_height, kernel_width, stride_h, stride_w)
        self.pos_bg = PositionEncoderRppg(in_channels,emb_size, 15, img_width, kernel_height, kernel_width, stride_h, stride_w)
        self.pos_final = PositionEncoderFinalRppg(in_channels,emb_size)
        self.transformer_6 = TransformerEncoder(depth, emb_size=emb_size,num_head = 3, **kwargs)
        self.transformer_1 = TransformerEncoder(1, emb_size=emb_size,num_head = 3, **kwargs)
        self.to_latent1 = nn.Identity()
        self.to_latent2 = nn.Identity()
        self.classificationFace = ClassificationHeadRppg(emb_size, n_classes)
        self.classificationBg = ClassificationHeadRppg(emb_size, n_classes)
        self.classification = ClassificationHeadRppg(emb_size, n_classes)

  def forward(self,x,y):
    face,bg = x,y
    # face = self.patch_embedded_face(face)
    # bg = self.patch_embedded_bg(bg)
    face = self.linear(face)
    bg= self.linear(bg)
    face = self.pos_face(face)
    bg = self.pos_bg(bg)
    face = self.transformer_6(face)
    bg = self.transformer_6(bg)
    # print(face.shape)
    # print(bg.shape)
    face = face[:,1:,:]
    face_token = face[:,0,:]
    bg = bg[:,1:,:]
    bg_token = bg[:,0,:]
    final = torch.cat((face, bg), 1)
    final = self.to_latent1(final)
    # print(final.shape)
    final = self.pos_final(final)
    # print(final.shape)
    final = self.transformer_1(final)
    # print(final.shape)
    final = final[:,0,:]
    final = self.to_latent2(final)
    final = self.classification(final)
    face_pred = self.classificationFace(face_token)
    bg_pred = self.classificationBg(bg_token)
    return final,face_pred,bg_pred


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransRppg2(nn.Module):
  def __init__(self,
               in_channels: int = 3,
               kernel_height = 3,
               kernel_width = 30,
               emb_size: int = 768,
               img_height: int = 63,
               img_width: int = 300,
               stride_h: int = 1,
               stride_w: int = 15,
                depth: int = 6,
                n_classes: int = 2,
                **kwargs):
        super().__init__()
        # self.patch_embedded_face = PatchEmbeddingTrans(in_channels,emb_size, img_height, img_width, kernel_height, kernel_width, stride_x, stride_y)
        # self.patch_embedded_bg = PatchEmbeddingTrans(in_channels,emb_size, 63, img_width, kernel_height, kernel_width, stride_x, stride_y)
        self.linear = LinearTrans(in_channels,emb_size, img_height, img_width, kernel_height, kernel_width, stride_h, stride_w)
        self.pos_face = PositionEncoderRppg(in_channels,emb_size, img_height, img_width, kernel_height, kernel_width, stride_h, stride_w)
        self.pos_bg = PositionEncoderRppg(in_channels,emb_size, 15, img_width, kernel_height, kernel_width, stride_h, stride_w)
        self.pos_final = PositionEncoderFinalRppg(in_channels,emb_size)
        # self.transformer_6 = TransformerEncoder(depth, emb_size=emb_size,num_head = 3, **kwargs)
        self.transformer_6 = Transformer(emb_size,depth,heads = 3,dim_head=emb_size,mlp_dim=emb_size)
        # self.transformer_1 = TransformerEncoder(1, emb_size=emb_size,num_head = 3, **kwargs)
        self.transformer_1 = Transformer(emb_size,1,heads = 3,dim_head=emb_size,mlp_dim=emb_size)
        self.to_latent1 = nn.Identity()
        self.to_latent2 = nn.Identity()
        self.classificationFace = ClassificationHeadRppg(emb_size, n_classes)
        self.classificationBg = ClassificationHeadRppg(emb_size, n_classes)
        self.classification = ClassificationHeadRppg(emb_size, n_classes)

  def forward(self,x,y):
    face,bg = x,y
    # face = self.patch_embedded_face(face)
    # bg = self.patch_embedded_bg(bg)
    face = self.linear(face)
    bg= self.linear(bg)
    face = self.pos_face(face)
    bg = self.pos_bg(bg)
    face = self.transformer_6(face)
    bg = self.transformer_6(bg)
    # print(face.shape)
    # print(bg.shape)
    face = face[:,1:,:]
    face_token = face[:,0,:]
    bg = bg[:,1:,:]
    bg_token = bg[:,0,:]
    final = torch.cat((face, bg), 1)
    final = self.to_latent1(final)
    # print(final.shape)
    final = self.pos_final(final)
    # print(final.shape)
    final = self.transformer_1(final)
    # print(final.shape)
    final = final[:,0,:]
    final = self.to_latent2(final)
    final = self.classification(final)
    face_pred = self.classificationFace(face_token)
    bg_pred = self.classificationBg(bg_token)
    return final,face_pred,bg_pred
