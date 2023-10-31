import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from utils import Linear, MatMul, Softmax, Clone
import math

#TODO: No está metida la clase TabAttention (porque creo que ni se usa, es SIMILAR a SAINT)
#TODO: faltan los helpers -> funciones (exists, default, ff_encodgings)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
    def relprop(self, cam, **kwargs):
        #TODO: revisar, se supone que la normalización se ignora para propagar la relevancia
        cam = self.fn.relprop(cam, **kwargs)
        return cam
        
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
    def relprop(self, cam, **kwargs):
        #TODO: revisar, se supone que lo residual se ignora para propagar la relevancia
        cam = self.fn.relprop(cam, **kwargs)
        return cam
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = Linear(dim, self.inner_dim * 3, bias = False)
        self.to_out = Linear(self.inner_dim, dim)

        self.dropout = nn.Dropout(dropout)
        
        #TODO: añadido nuevo en PRUEBAS
        self.query = Linear(dim, self.inner_dim)
        self.key = Linear(dim, self.inner_dim)
        self.value = Linear(dim, self.inner_dim)
        self.matmul1 = MatMul()
        self.matmul2 = MatMul()
        self.softmax = Softmax(dim=-1)
        self.clone = Clone()

        #Añadido nuevo
        self.attn_cam = None
        self.attn = None
        self.attn_gradients = None
    
    def get_attn(self):
        return self.attn
    
    def save_attn(self, attn):
        self.attn = attn

    def get_attn_cam(self):
        return self.attn_cam
    
    def save_attn_cam(self, cam):
        self.attn_cam = cam
    
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
    
    #TODO: nuevo añadido
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.dim_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    #TODO: nuevo añadido
    def transpose_for_scores_relprop(self, x):
        return x.permute(0, 2, 1, 3).flatten(2)

    def forward(self, x):
        """h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        attn_cal = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn_prob = attn_cal.softmax(dim = -1)

        self.save_attn(attn_prob)
        if self.training != True and x.requires_grad != False:
            attn_prob.register_hook(self.save_attn_gradients)

        out = einsum('b h i j, b h j d -> b h i d', attn_prob, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        output = self.to_out(out)"""
    
        #TODO: segunda forma de calcularlo
        h1, h2, h3 = self.clone(x, 3)
        query_layer = self.transpose_for_scores(self.query(h1))
        key_layer = self.transpose_for_scores(self.key(h2))
        value_layer = self.transpose_for_scores(self.value(h3))

        attention_scores = self.matmul1([query_layer, key_layer.transpose(-1, -2)])
        attention_scores = attention_scores / math.sqrt(self.dim_head)

        attention_probs = self.softmax(attention_scores)

        self.save_attn(attention_probs)
        if self.training != True and x.requires_grad != False:
            attention_probs.register_hook(self.save_attn_gradients)
        
        context_layer = self.matmul2([attention_probs, value_layer])
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.inner_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.to_out((context_layer,)[0])

        return output


    def relprop(self, cam=None, **kwargs):
        #TODO
        cam = self.to_out.relprop(cam, **kwargs) #añadido nuevo
        cam = self.transpose_for_scores(cam)
        (cam1, cam2) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam2 /= 2
        #if self.head_mask is not None:
            # [attention_probs, head_mask]
            #(cam1, _)= self.mul.relprop(cam1, **kwargs)

        self.save_attn_cam(cam1)

        #cam1 = self.dropout.relprop(cam1, **kwargs)

        cam1 = self.softmax.relprop(cam1, **kwargs)

        #if self.attention_mask is not None:
            # [attention_scores, attention_mask]
            #(cam1, _) = self.add.relprop(cam1, **kwargs)

        (cam1_1, cam1_2) = self.matmul1.relprop(cam1, **kwargs)
        cam1_1 /= 2
        cam1_2 /= 2

        cam1_1 = self.transpose_for_scores_relprop(cam1_1)
        cam1_1 = self.query.relprop(cam1_1, **kwargs)

        cam1_2 = self.transpose_for_scores_relprop(cam1_2.transpose(-1, -2))
        cam1_2 = self.key.relprop(cam1_2, **kwargs)

        cam2 = self.transpose_for_scores_relprop(cam2)
        cam2 = self.value.relprop(cam2, **kwargs)

        cam = self.clone.relprop((cam1_1, cam1_2, cam2), **kwargs)

        return cam


#TODO: nuevo añadido
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
    
    def relprop(self, cam, **kwargs):
        #TODO
        last_size = cam.shape
        cam = torch.cat((cam, cam), 2)
        mask = torch.zeros(cam.shape).to(cam.device)
        mask[:, :, :last_size[2]] = 1
        cam = torch.mul(cam, mask)
        return cam

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

    def relprop(self, cam=None, **kwargs):
        #TODO
        layers = self.net._modules
        for i, layer in reversed(layers.items()):
            if(layer.__class__.__name__ == "Linear" or layer.__class__.__name__ == "GEGLU"):
                cam = layer.relprop(cam, **kwargs)
        return cam


"""class SAINT_Transformer_layer(nn.Module):
    def __init__(self, dim, nfeats, heads, dim_head, attn_dropout, ff_dropout, style):
        super().__init__()
        if style == 'colrow':
            PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
            PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
            PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
            PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
        
        elif style == 'row':
            PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
            PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),


        elif style == 'col':
            PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
            PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
        
        else:
            print("errorrrr") #TODO"""



#################################################################

class SAINT_Transformer(nn.Module):
    #TODO: revisar
    #variables en común: num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout
    #solo de RowCol: nfeats, style='col'


    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col'):
        super().__init__()
        
        # asignamos los valores a las variables
        self.style = style
        self.layers = nn.ModuleList([])

        #TODO: ESTABA EN COLROW Y ROW
        #self.embeds = nn.Embedding(num_tokens, dim)
        #self.mask_embed =  nn.Embedding(nfeats, dim)

        # definimos la estructura del modelo (SAINT, SAINT-s o SAINT-i)
        #TODO:#self.layers = nn.ModuleList([SAINT_Transformer_layer(dim, nfeats, heads, dim_head, attn_dropout, ff_dropout, style) for _ in range(depth)])

        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))
        
            elif self.style == 'row':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))

            elif self.style == 'col':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                ]))
            
            else:
                print("ERROR - ")

    
    def forward(self, x, x_cont=None, mask = None): #TODO: mask no se utiliza, REVISAR
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers: 
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        elif self.style == 'row':
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        elif self.style == 'col':
            for attn1, ff1 in self.layers:
                x = attn1(x)
                x = ff1(x)
        else:
            print("ERROR - ")
        return x

    def relprop(self, cam=None, **kwargs):
        #TODO
        #va en orden inverso, primero relprop de ffn y luego de attn
        _, n, _ = cam.shape
        if self.style == 'colrow':   
            for attn1, ff1, attn2, ff2 in self.layers: 
                cam = rearrange(cam, 'b n d -> 1 b (n d)')
                cam = ff2.relprop(cam, **kwargs)
                cam = attn2.relprop(cam, **kwargs)
                cam = rearrange(cam, '1 b (n d) -> b n d', n = n)
                cam = ff1.relprop(cam, **kwargs)
                cam = attn1.relprop(cam, **kwargs)
        elif self.style == 'row':
            #TODO
            print("row")
        elif self.style == 'col':
            #TODO
            print("col")
        else:
            print("ERROR - ")

        return cam