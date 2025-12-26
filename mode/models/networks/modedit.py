from typing import Optional
import logging
import math 

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


class FusedExpertCache:
    """Simple cache storing pre-fused expert weights"""
    def __init__(self):
        self.fused_weights = {}  # {layer_idx: {noise_level: FusedMLP}}
        self.routing_info = {}   # For debugging/analysis 

class FusedMLPV2(nn.Module):
    def __init__(self, expert1: nn.Module, expert2: nn.Module, probs=None):
        super().__init__()
        self.mlp1 = expert1
        self.mlp2 = expert2
        self.probs = probs if not None else [0.5, 0.5]
        in_dim = self.mlp1.mlp[0].project.in_features
        hidden_dim = self.mlp1.mlp[0].project.out_features
        out_dim = self.mlp1.mlp[2].out_features

        self.w1 = nn.Linear(in_dim, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

        # create a fused version of the weights of the two MLPs
        self.w1.weight.data = torch.cat(
            [
                self.mlp1.mlp[0].project.weight.data,
                self.mlp2.mlp[0].project.weight.data,
            ],
            dim=0,
        )
        self.w1.bias.data = torch.cat(
            [self.mlp1.mlp[0].project.bias.data, self.mlp2.mlp[0].project.bias.data],
            dim=0,
        )
        self.w2.weight.data = torch.cat(
            [self.mlp1.mlp[2].weight.data, self.mlp2.mlp[2].weight.data], dim=1
        )
        self.act = nn.SiLU()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Reshape to 2D for linear layer
        x = x.reshape(-1, dim)
        
        # Apply same operations as 2D version
        x = self.w1(x).tensor_split(4, dim=-1)
        stack1 = torch.stack([x[1], x[3]])
        stack2 = torch.stack([x[0]*self.probs[0], x[2]*self.probs[1]])
        
        activated = self.act(stack1)
        result = (activated * stack2).transpose(0, 1).reshape(batch_size * seq_len, self.hidden_dim)
        result = self.w2(result)
        
        # Reshape back to 3D
        return result.reshape(batch_size, seq_len, -1)
    

# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g
    
# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)



class Attention(nn.Module):
    def __init__(
        self, 
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        causal: bool = False,
        bias=False,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and causal:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
        # Dynamically compute causal mask instead of using a fixed bias buffer
        self.block_size = block_size
        self.qk_norm = qk_norm
        # init qk norm if enabled
        if self.qk_norm:
            self.q_norm = RMSNorm(n_embd//self.n_head, eps=1e-6)
            self.k_norm = RMSNorm(n_embd//self.n_head, eps=1e-6)
        else: 
            self.q_norm = self.k_norm = nn.Identity()



    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size()

        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=custom_attn_mask, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Optimize custom attention masking
            if custom_attn_mask is not None:
                att = att.masked_fill(custom_attn_mask == 0, float('-inf'))
            elif self.causal:
                # Dynamically compute causal mask based on current sequence length T
                causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
                att = att.masked_fill(causal_mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class CondRouterMLP(nn.Module):

    def __init__(
            self, 
            n_embd: int,
            num_experts: int,
            use_swish: bool = True,
            use_relus: bool = False,
            dropout: float = 0,
            make_it_big: bool = False
        ):
        super().__init__()
        layers = []
        factor = 2 if make_it_big else 1  # Factor to double the hidden dimensions if make_it_big is True
        repeat = 2 if make_it_big else 1  # Repeat layers if make_it_big is True

        for i in range(repeat):  # Repeat constructing layers to double the number of layers if make_it_big
            if i == 0:
                curr_embed = n_embd
            else:
                curr_embed = factor * 2 * n_embd
            if use_swish:
                layers.append(SwishGLU(curr_embed, factor * 2 * n_embd))
            else:
                layers.append(nn.Linear(curr_embed, factor * 2 * n_embd))
                if use_relus:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # Ensure that the last layer always maps to the number of experts
        layers.append(nn.Linear(factor * 2 * n_embd, num_experts))
        
        self.mlp = nn.Sequential(*layers)

        # Initialize weights with zeros 
        self._init_weights()

    def forward(self, x):
        return self.mlp(x)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Mlp(nn.Module):
    def __init__(
            self, 
            n_embd: int,
            bias: bool,
            use_swish: bool = True,
            use_relus: bool = False,
            dropout: float = 0,
            identity_only: bool = False,
            output_dim: Optional[int] = None
        ):
        super().__init__()
        self.identity_only = identity_only
        layers = []

        if output_dim is not None:
            n_embed_final = output_dim
        else:
            n_embed_final = n_embd
        
        if identity_only:
            # Initialize as identity layers
            identity_layer = nn.Linear(n_embd, n_embd, bias=False)
            nn.init.eye_(identity_layer.weight)  # Set weights to identity matrix
            layers.append(identity_layer)
        else:
            if use_swish:
                layers.append(SwishGLU(n_embd, 4 * n_embd))
            else:
                layers.append(nn.Linear(n_embd, 4 * n_embd, bias=bias))
                if use_relus:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(4 * n_embd, n_embed_final, bias=bias))
        
        self.mlp = nn.Sequential(*layers)
        
        if identity_only:
            # Freeze the parameters so they are not updated during training
            for param in self.mlp.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.mlp(x)
    


class RouterCond(nn.Module):
    def __init__(
        self,
        hidden_states: int,
        cond_dim: int,
        num_experts: int,
        top_k: int,
        use_argmax: bool = False,
        normalize: bool = True,
        cond_router: bool = True,
        router_context_cond_only: bool = False,
        temperature: float = 1.0,
        use_shared_expert: bool = False,
    ):
        """Initialize the RouterCond module."""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize = normalize
        self.temperature = temperature
        self.use_argmax = use_argmax
        self.use_shared_expert = use_shared_expert
        self.cond_router = cond_router
        self.router_context_cond_only = router_context_cond_only

        self.router = self._create_router(hidden_states, cond_dim)
        self.logits = None

    def _create_router(self, hidden_states: int, cond_dim: int) -> nn.Module:
        """Create the router MLP based on the configuration."""
        if self.cond_router:
            input_dim = cond_dim if self.router_context_cond_only else hidden_states + cond_dim
        else:
            input_dim = hidden_states

        return CondRouterMLP(
            input_dim,
            self.num_experts,
            use_swish=False,
            dropout=0,
            make_it_big=False
        )

    def forward(self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """Forward pass of the router."""
        input_shape = inputs.size()
        logits = self._compute_logits(inputs, cond)
        probs = self._compute_probabilities(logits)
        router_mask, top_k_indices, router_probs = self._select_experts(probs, input_shape)
        return router_mask, top_k_indices, router_probs, probs.view(*input_shape[:-1], -1)

    def _compute_logits(self, inputs: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute logits based on inputs and conditional information."""
        if self.cond_router:
            return self._compute_cond_logits(inputs, cond)
        return self._compute_uncond_logits(inputs)

    def _compute_cond_logits(self, inputs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute logits for conditional routing."""
        if cond.shape[-2] != inputs.shape[-2]:
            cond = einops.repeat(cond, 'b t d -> b (t n) d', n=int(inputs.shape[-2] / cond.shape[-2]))

        if self.router_context_cond_only:
            router_inputs = cond.reshape(-1, cond.size(-1))
        else:
            router_inputs = torch.cat([inputs, cond], dim=-1).reshape(-1, inputs.size(-1) + cond.size(-1))

        logits = self.router(router_inputs)
        return logits

    def _compute_uncond_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute logits for unconditional routing."""
        return self.router(inputs.reshape(-1, inputs.size(-1)))

    def _compute_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute probabilities from logits."""
        logits = (logits - logits.max(dim=-1, keepdim=True).values) / self.temperature
        self.logits = logits

        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-9, max=1-1e-9)

        self._validate_probabilities(probs)
        return probs

    def _validate_probabilities(self, probs: torch.Tensor):
        """Validate the computed probabilities."""
        if not torch.isfinite(probs).all():
            logging.warning("Probabilities contain inf or NaN values")
        if not torch.allclose(probs.sum(dim=-1), torch.tensor(1.0, dtype=probs.dtype), atol=1e-5):
            logging.warning("Probabilities do not sum up to 1")

    def _select_experts(self, probs: torch.Tensor, input_shape: torch.Size):
        """Select experts based on computed probabilities."""
        if self.use_shared_expert and self.top_k == 2:
            return self._select_experts_with_shared(probs, input_shape)
        return self._select_experts_without_shared(probs, input_shape)

    def _select_experts_with_shared(self, probs: torch.Tensor, input_shape: torch.Size):
        """Select experts when using a shared expert."""
        shared_expert_index = self.num_experts - 1
        other_probs = probs[:, :3]
        other_expert_index = torch.multinomial(other_probs, 1) if self.training and not self.use_argmax else other_probs.topk(1, dim=-1).indices
        
        top_k_indices = torch.cat([other_expert_index, torch.full_like(other_expert_index, shared_expert_index)], dim=-1)
        router_mask = torch.zeros_like(probs).scatter_(1, top_k_indices, 1)
        
        router_probs = probs.clone()
        router_probs[:, 3:shared_expert_index] = 0
        router_probs = router_probs * router_mask

        return self._format_output(router_mask, top_k_indices, router_probs, input_shape)

    def _select_experts_without_shared(self, probs: torch.Tensor, input_shape: torch.Size):
        """Select experts when not using a shared expert."""
        # print(f"Debug - _select_experts_without_shared input shapes: probs {probs.shape}, input_shape {input_shape}")
        
        # Flatten batch dimensions
        flat_probs = probs.view(-1, probs.size(-1))
        
        if self.training and not self.use_argmax:
            top_k_indices = torch.multinomial(flat_probs, self.top_k, replacement=False)
        else:
            top_k_indices = flat_probs.topk(self.top_k, dim=-1).indices
        
        # print(f"Debug - top_k_indices shape: {top_k_indices.shape}, device: {top_k_indices.device}")
        # print(f"Debug - top_k_indices min: {top_k_indices.min()}, max: {top_k_indices.max()}")
        
        try:
            router_mask = torch.zeros_like(flat_probs).scatter_(1, top_k_indices, 1)
            router_probs = torch.zeros_like(flat_probs).scatter_(1, top_k_indices, flat_probs.gather(1, top_k_indices))
            
            # Reshape back to original dimensions
            router_mask = router_mask.view(probs.shape)
            router_probs = router_probs.view(probs.shape)
            top_k_indices = top_k_indices.view(probs.shape[:-1] + (self.top_k,))
        except RuntimeError as e:
            print(f"Error in scatter_ operation: {e}")
            print(f"Debug - flat_probs shape: {flat_probs.shape}, top_k_indices shape: {top_k_indices.shape}")
            print(f"Debug - flat_probs device: {flat_probs.device}, top_k_indices device: {top_k_indices.device}")
            raise
        return self._format_output(router_mask, top_k_indices, router_probs, input_shape)

    def _format_output(self, router_mask: torch.Tensor, top_k_indices: torch.Tensor, router_probs: torch.Tensor, input_shape: torch.Size):
        """Format the output of the expert selection process."""
        router_mask = router_mask.view(*input_shape[:-1], -1)
        top_k_indices = top_k_indices.view(*input_shape[:-1], -1)
        router_probs = router_probs.view(*input_shape[:-1], -1)

        if self.normalize:
            router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)

        return router_mask, top_k_indices, router_probs
    

class NoiseBlockMoE(nn.Module):
    """
    Block with AdaLN-Zero conditioning and efficient expert caching.
    """
    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            mlp_pdrop: float, 
            noise_in_cross_attention: bool = False,
            cond_router: bool = False,
            use_cross_attention: bool = False, 
            num_experts: int = 4,
            top_k: int = 2,
            router_normalize: bool = True,
            router_context_cond_only: bool = True,
            use_argmax: bool = False,
            use_shared_expert: bool = False,
            identity_expert: bool = False,
            attn_arg: str = 'causal',
        ):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd, eps=1e-6)
        self.n_embd = n_embd
        self.attn = Attention(
            n_embd, 
            n_heads, 
            qk_norm=True,
            attn_pdrop=attn_pdrop,
            resid_pdrop=0,
            block_size=100,
            causal=True,
        )
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = Attention(
                n_embd, 
                n_heads, 
                qk_norm=True,
                attn_pdrop=attn_pdrop,
                resid_pdrop=0,
                causal=True,
            )
            self.ln_3 = RMSNorm(n_embd, eps=1e-6) 

        self.ln_2 = RMSNorm(n_embd, eps=1e-6) 
        self.logits = None
        
        self.cond_router = cond_router
        self.num_experts = num_experts
        self.use_shared_expert = use_shared_expert

        if self.use_shared_expert:
            top_k_router = top_k - 1
            num_experts_router = num_experts - 1
        else:
            num_experts_router = num_experts
            top_k_router = top_k

        self.router = RouterCond(
            n_embd, 
            n_embd,
            num_experts_router, 
            top_k_router, 
            use_argmax=use_argmax,
            normalize=router_normalize,
            cond_router=cond_router,
            router_context_cond_only=router_context_cond_only,
        )

        self.experts = nn.ModuleDict(
            {
                f"expert_{i}": Mlp(
                    n_embd,  # in_features
                    bias=False,
                    dropout=mlp_pdrop
                )
                for i in range(num_experts_router - int(identity_expert))
            }
        )
        if self.use_shared_expert:
            self.shared_mlp = Mlp(n_embd, bias=False, dropout=mlp_pdrop)

        if identity_expert:
            self.experts[f"expert_{num_experts_router}"] = nn.Identity()
        
        # Cache for expert combinations during inference
        self.expert_cache = {}

        self.noise_in_cross_attention = noise_in_cross_attention
        self.probs = None
        self.top_k = None
        
        # To track the usage of each expert
        self.expert_usage = torch.zeros(num_experts_router)
        self.train_expert_usage = torch.zeros(num_experts_router)
        # Initialize expert usage tracking
        num_experts_router = num_experts - 1 if use_shared_expert else num_experts
        self.expert_usage = torch.zeros(num_experts_router)
        self.inference_expert_usage = torch.zeros(num_experts_router)
        self.total_tokens_processed = 0

        self.fused_experts = {}  # {noise_level: FusedMLP}
        self.routing_info = {}   # For debugging/analysis

    def forward(self, x, c, context=None, custom_attn_mask=None):
        # First apply attention
        x = x + self.attn(self.ln_1(x) + c, custom_attn_mask=custom_attn_mask)
        
        if self.use_cross_attention and context is not None:
            if self.noise_in_cross_attention:
                x = x + self.cross_att(self.ln_3(x) + c, context, custom_attn_mask=custom_attn_mask)
            else:
                x = x + self.cross_att(self.ln_3(x), context, custom_attn_mask=custom_attn_mask)
        x = self.ln_2(x)

        # Check if we're in inference mode and have precomputed experts
        if not self.training and self.fused_experts:  # Only use cache if it's not empty
            noise_key = c.mean().item()  # or however you want to get the noise key
            if noise_key in self.fused_experts:
                fused_mlp = self.fused_experts[noise_key]
                return x + fused_mlp(x)
            
        batch_tokens = x.size(0) * x.size(1)
        # Normal forward pass (for training or when cache is empty)
        if self.cond_router:
            router_mask, top_k_indices, router_probs, true_probs = self.router(x, c)
        else:
            router_mask, top_k_indices, router_probs, true_probs = self.router(x, None)
        
        next_states = torch.zeros_like(x)

        # Track expert usage for both training and validation when not using cache
        num_balanced_experts = len(self.experts)
        
        # Process inputs through selected experts
        for idx in range(num_balanced_experts):
            token_indices = router_mask[:, :, idx].bool()
            if token_indices.any():
                expert = self.experts[f"expert_{idx}"]
                probs = router_probs[:, :, idx][token_indices].unsqueeze(-1)
                next_states[token_indices] += probs * expert(x[token_indices]).to(next_states.dtype)

                # Track expert usage statistics
                if self.training:
                    self.expert_usage[idx] += token_indices.sum().item()
                else:
                    self.inference_expert_usage[idx] += token_indices.sum().item()

        # Apply shared expert if enabled
        if self.use_shared_expert:
            shared_output = self.shared_mlp(x)
            if router_mask.size(-1) > num_balanced_experts:
                shared_prob = router_probs[:, :, -1].unsqueeze(-1)
                next_states += shared_prob * shared_output
            else:
                next_states = next_states + shared_output

        # Store routing information during training
        if self.training:
            self.logits = self.router.logits
            self.probs = {
                "probs": true_probs,
                "top_k_hot": router_mask,
                "load_balancing_term": num_balanced_experts * (
                    router_probs[:, :, :num_balanced_experts].mean(dim=(0, 1)) * 
                    (router_mask[:, :, :num_balanced_experts].sum(dim=(0, 1)) / batch_tokens)
                ).sum()
            }
        self.total_tokens_processed += batch_tokens
        return x + next_states

    def get_expert_usage(self):
        """Get the combined expert usage statistics"""
        return self.inference_expert_usage

    def reset_expert_usage(self):
        """Reset all expert usage statistics"""
        self.expert_usage.zero_()
        self.inference_expert_usage.zero_()
        self.total_tokens_processed = 0

    def prepare_fused_expert(self, cond_embedding):
        """Prepare fused expert for a noise level"""
        if self.training:
            return

        with torch.no_grad():
            noise_key = cond_embedding.mean().item()
            dummy_input = torch.zeros((1, 1, self.n_embd), device=cond_embedding.device)
            
            # Get router decisions
            _, top_k_indices, router_probs, _ = self.router(dummy_input, cond_embedding)
            
            # Get top 2 experts and their probabilities
            indices = top_k_indices[0, 0][:2]  # Take first 2 indices
            probs = router_probs[0, 0, indices]
            
            # Create fused expert from top 2 
            expert1 = self.experts[f"expert_{indices[0]}"]
            expert2 = self.experts[f"expert_{indices[1]}"] if indices[1] < len(self.experts) else self.shared_mlp
            
            self.fused_experts[noise_key] = FusedMLPV2(expert1, expert2, probs=[probs[0].item(), probs[1].item()])
            
            # Store routing info for debugging
            self.routing_info[noise_key] = {
                'indices': indices.cpu().numpy(),
                'probs': probs.cpu().numpy()
            }

    def reset_expert_cache(self):
        """Reset expert cache"""
        self.fused_experts = {}
    


class MoDeDiT(nn.Module):

    def __init__(
        self, 
        obs_dim: int,
        goal_dim: int,
        device: str,
        goal_conditioned: bool,
        action_dim: int,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        state_dim,
        mlp_pdrop: float = 0.1,
        goal_drop: float = 0.1,
        linear_output: bool = True,
        use_proprio: bool = False,
        cond_router: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        router_normalize: bool = True,
        use_goal_in_routing: bool = False,
        use_argmax: bool = False,
        causal: bool = True,
        use_shared_expert: bool = False,
        use_noise_token_as_input: bool = True,
        use_custom_attn_mask: bool = False,
        init_style: str = 'default'
    ):
        super().__init__()
        self.device = device
        self.use_proprio = use_proprio
        self.obs_dim = obs_dim
        # self.sigma_emb =  NoiseEmbedding(embed_dim)
        self.sigma_emb = nn.Linear(1, embed_dim)
        self.sigma_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len
        self.tok_emb = nn.Linear(obs_dim, embed_dim, bias=False)
        self.gripper_embed = nn.Linear(obs_dim, embed_dim, bias=False)
        self.goal_emb = nn.Linear(goal_dim, embed_dim, bias=False)
        self.action_emb = nn.Linear(action_dim, embed_dim, bias=False)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        if self.use_proprio:
            self.state_embed = nn.Linear(state_dim, embed_dim, bias=False)
        self.cond_mask_prob = goal_drop
        self.blocks = nn.ModuleList([])
        self.num_layers = n_layers
        self.embed_dim = embed_dim

        if use_custom_attn_mask:
            self.use_custom_attn_mask = True
            attn_arg = 'custom'
        else:
            self.use_custom_attn_mask = False
            attn_arg = 'causal'

        for _ in range(n_layers):
            self.blocks.append(
                NoiseBlockMoE(
                    embed_dim, 
                    n_heads, 
                    attn_pdrop, 
                    mlp_pdrop, 
                    mlp_pdrop,  
                    cond_router=cond_router,
                    num_experts=num_experts,
                    top_k=top_k,
                    router_normalize=router_normalize,
                    use_shared_expert=use_shared_expert,
                    use_argmax=use_argmax,
                    attn_arg=attn_arg,
                )
            )
        self.ln = RMSNorm(embed_dim, eps=1e-6)
        self.linear_output = linear_output
        if self.linear_output:
            self.out = nn.Linear(embed_dim, action_dim)
        else:
            self.out = Mlp(embed_dim, bias=False, dropout=mlp_pdrop)

        self.goal_seq_len = goal_seq_len
        self.action_seq_len = action_seq_len
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.use_noise_token_as_input = use_noise_token_as_input
        self.use_goal_in_routing = use_goal_in_routing
        self.init_style = init_style
        self.goal_conditioned = goal_conditioned
        self.causal = causal
        logger.info("Weights initialized using custom _init_weights method")
        self.logits_per_layer = None
        self.probs_per_layer = None

    def forward(
        self, 
        states,
        actions, 
        goals,
        sigma,
        uncond: Optional[bool] =False,
    ):  
        t = 1
        # Process sigma embeddings
        emb_t = self.process_sigma_embeddings(sigma)
        
        # Reshape goals if necessary
        goals = self.preprocess_goals(goals, 1, uncond=uncond)

        # embed them into linear representations for the transformer
        if len(goals.shape) == 2:
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        
        state_embed = self.tok_emb(states['state_images'])
        if 'robot_obs' in states and self.use_proprio:
            proprio_embed = self.process_state_obs(states['robot_obs'].to(goals.dtype))
        else:
            proprio_embed = None
        goal_embed = self.goal_emb(goals)
        action_embed = self.action_emb(actions)
        
        # if not uncond:
        if self.goal_conditioned:
            position_embeddings = self.pos_emb[
            :, :(t + self.goal_seq_len + self.action_seq_len - 1), :
            ]  # each position maps to a (learnable) vector
        else: # without goal conditioning we only have the obs sequence 
            position_embeddings = self.pos_emb[
                :, :t, :
            ]
        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the causal masking would not make sense
        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len+t), :])
        # the action get the same position embedding as the related states 
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len+t-1):, :])
        if 'robot_obs' in states and self.use_proprio:
            proprio_x = self.drop(proprio_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len+t)])
        else:
            proprio_x = None
        # next we stack everything together 
        # if use_noise_token_as_input is True we add a noise token to the input sequence
        # if goal_conditioned is False we only have the obs sequence
        input_seq = self.build_input_seq(state_x, action_x, goal_x, emb_t, proprio_x)

        # input_seq = self.mask_cond(input_seq)

        if self.use_custom_attn_mask:
            custom_mask = self.create_custom_mask(input_seq.shape[1])
        else:
            custom_mask = None

        cond_token = emb_t
        
        if self.use_goal_in_routing:
            cond_token = cond_token + goal_embed
        # Note we need to also adapt the action masks 
        x = self.forward_modedit(input_seq, cond_token, custom_attn_mask=custom_mask)
        # x = self.ln_f(x)
        # now we want the last half of the output      
        action_outputs =x[:, -self.action_seq_len:, :]
        pred_actions = self.out(action_outputs)
        return pred_actions
    
    def forward_modedit(self, x, c, custom_attn_mask=None):
        logits_per_layer = []
        probs_per_layer = []
        for layer in self.blocks:
            x = layer(x, c, c, custom_attn_mask=custom_attn_mask)
            logits_per_layer.append(layer.logits)
            probs_per_layer.append(layer.probs)
        x = self.ln(x)
        self.logits_per_layer = logits_per_layer
        self.probs_per_layer = probs_per_layer
        return x

    def process_sigma_embeddings(self, sigma):
        sigmas = sigma.log() / 4 # log-normalize sigma
        if sigmas.dim() == 0:
            sigmas = sigmas.unsqueeze(0)
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas)
        emb_t = self.sigma_linear(emb_t)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        return emb_t

    def process_state_obs(self, state_obs):
        # split into prior and gripper state
        proprio = state_obs[:, :, :-2]
        gripper_state = state_obs[:, :, -2:]

        # encode proprio
        proprio_emb = self.tok_emb(proprio)
        gripper_emb = self.gripper_embed(gripper_state)

        combined_embed = self.combine_embed(torch.cat([proprio_emb, gripper_emb], dim=-1))

        return combined_embed

    def build_input_seq(self, state_x, action_x, goal_x=None, emb_t=None, proprio_embed=None):
        sequences = []
        if self.use_noise_token_as_input and emb_t is not None:
            sequences.append(emb_t)

        if self.goal_conditioned and goal_x is not None:
            sequences.append(goal_x)

        if proprio_embed is not None:
            sequences.append(proprio_embed)
        
        sequences.extend([state_x, action_x])
                
        return torch.cat(sequences, dim=1)

    def preprocess_goals(self, goals, states_length, uncond=False):

        if len(goals.shape) == 2:
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[1] == states_length and self.goal_seq_len == 1:
            goals = goals[:, 0, :]
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[-1] == 2 * self.obs_dim:
            goals = goals[:, :, :self.obs_dim]
        # during training randomly mask out the goal
        # to train the conditional model with classifier-free guidance wen need 
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2 
        if self.training:
            goals = self.mask_cond(goals)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)  
        return goals
    
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # TODO Check which one is correct
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob) # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            # mask = torch.bernoulli(torch.ones((bs, t, 1), device=cond.device) * self.cond_mask_prob)
            # mask = einops.repeat(mask, 'b t 1 -> b t (1 d)', d=d)
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()
    
    def load_balancing_loss(self, ):
        """ 
        Compute the load balancing loss for MoE with separate control for entropy and KL divergence.
        
        Args:
            probs: List of dictionaries, each containing "probs" and "top_k_hot" tensors
            use_entropy: Boolean to include entropy term in the loss
            use_kl: Boolean to include KL divergence term in the loss
            entropy_weight: Weight for the entropy term
            kl_weight: Weight for the KL divergence term
            balance_weight: Weight for the original balance term
        Returns:
            Scalar loss value
        """
        total_loss = 0.0

        if 'load_balancing_loss' not in self.probs_per_layer[0]:
            self.probs_per_layer[0]['load_balancing_loss'] = []
            for layer in self.blocks:
                if hasattr(layer, 'probs') and 'load_balancing_term' in layer.probs:
                    self.probs_per_layer[0]['load_balancing_loss'].append(layer.probs['load_balancing_term'])

        list_of_losses = self.probs_per_layer[0]['load_balancing_loss']

        for block_loss in list_of_losses:
            total_loss += block_loss
        
        if len(list_of_losses) > 0:
            total_loss = total_loss / len(list_of_losses)

        return total_loss
    
    def compute_router_z_loss(self, eps=1e-6):
        """
        Compute the router z-loss.
        
        Args:
        router_logits (torch.Tensor): The logits from the router, shape [batch_size, sequence_length, num_experts]
        eps (float): A small constant for numerical stability
        
        Returns:
        torch.Tensor: The computed z-loss
        """
        router_logits = self.logits_per_layer
        total_z_loss = 0
        num_layers = len(router_logits)
        
        for router_logits in router_logits:
            # print(router_logits.shape)
            B, S = router_logits.shape
            
            # Compute exp(x_j) for all logits
            exp_logits = torch.exp(router_logits)
            
            # Sum across the number of experts
            sum_exp = torch.sum(exp_logits, dim=-1)  # Shape: [B, S]
            
            # Compute log(sum(exp(x_j)))
            log_sum_exp = torch.log(sum_exp + eps)  # Shape: [B, S]
            
            # Square the result
            squared_log_sum_exp = log_sum_exp ** 2  # Shape: [B, S]
            
            # Average across the sequence length and then across the batch
            layer_z_loss = torch.mean(squared_log_sum_exp)
            
            total_z_loss += layer_z_loss
        
        # Compute the average z-loss across all layers
        average_z_loss = total_z_loss / num_layers
        
        return average_z_loss

    def precompute_experts_for_inference(self, sigma, goal=None):
        """
        Precompute experts for each noise level.
        """
        if self.training:
            return

        # Process noise embeddings
        emb_t = self.process_sigma_embeddings(sigma)
        
        if self.use_goal_in_routing and goal is not None:
            goal_embed = self.goal_emb(goal)
            cond_token = emb_t + goal_embed
        else:
            cond_token = emb_t

        # Precompute for each block
        for block in self.blocks:
            try:
                block.prepare_fused_expert(cond_token)
            except Exception as e:
                print(f"Error precomputing experts: {e}")

    def reset_all_caches(self):
        """Reset all expert caches"""
        for block in self.blocks:
            block.reset_expert_cache()

    def freeze_router(self):
        """Freeze router weights and set to eval mode"""
        for layer in self.blocks:
            layer.router.eval()
            for param in layer.router.parameters():
                param.requires_grad = False

    def unfreeze_router(self):
        """Unfreeze router weights and set to train mode"""
        for layer in self.blocks:
            layer.router.train()
            for param in layer.router.parameters():
                param.requires_grad = True

    def get_router_states(self):
        """Get current router training states for monitoring"""
        states = []
        for i, layer in enumerate(self.blocks):
            states.append({
                'layer': i,
                'frozen': not any(p.requires_grad for p in layer.router.parameters()),
                'eval_mode': not layer.router.training,
                'cache_size': len(layer.router.expert_cache)
            })
        return states

    def prepare_for_finetuning(
            self, 
            freeze_routers: bool = True,
            freeze_expert_weights: float = 0.3,
            reset_expert_stats: bool = True
        ):
        """Prepare model for finetuning by selectively freezing components
        
        Args:
            freeze_routers: Whether to freeze router weights
            freeze_expert_weights: Fraction of expert weights to freeze (0-1)
            reset_expert_stats: Whether to reset expert usage statistics
        """
        if freeze_routers:
            self.freeze_router()
            
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if self.init_style == 'switch_t':
            if isinstance(module, (nn.Linear, nn.Embedding)):
                fan_in = module.weight.size(1)
                scale = 0.1  # As per the previous recommendation
                std = math.sqrt(scale / fan_in)
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
            elif isinstance(module, MoDeDiT):
                fan_in = module.pos_emb.size(1)
                scale = 0.1
                std = math.sqrt(scale / fan_in)
                torch.nn.init.trunc_normal_(module.pos_emb, mean=0.0, std=std, a=-2*std, b=2*std)
        elif self.init_style == 'olmoe':
            if isinstance(module, (nn.Linear, nn.Embedding)):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
            elif isinstance(module, MoDeDiT):
                torch.nn.init.trunc_normal_(module.pos_emb, mean=0.0, std=0.02, a=-0.04, b=0.04)
        else:  # default initialization
            if isinstance(module, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                if hasattr(module, 'out_features') and module.out_features == self.num_experts:
                    torch.nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
            elif isinstance(module, MoDeDiT):
                torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)