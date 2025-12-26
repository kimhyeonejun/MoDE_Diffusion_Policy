"""
Model definitions for compression + VLA training.
Contains adapter models, loss functions, and custom dataset classes.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from typing import Dict
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
from transformers import AutoProcessor
import clip
import CLIP_modify.clip as clip_modify


class CombinedIterableDataset(IterableDataset):
    """Combines multiple iterable datasets by randomly sampling from them."""
    
    def __init__(self, *iterable_datasets):
        self.iterable_datasets = iterable_datasets

    def __iter__(self):
        iterators = [iter(ds) for ds in self.iterable_datasets]
        
        while iterators:
            rand_idx = random.randrange(len(iterators))
            chosen_iterator = iterators[rand_idx]
            
            try:
                yield next(chosen_iterator)
            except StopIteration:
                iterators.pop(rand_idx)


class Linear_Encoder(nn.Module):
    """Simple linear adapter for image feature compression."""
    
    def __init__(self, in_features, out_features, input_resolution, args):
        super(Linear_Encoder, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class InstructionalClipLoss(nn.Module):
    """
    CLIP-based loss using dynamic task instructions.
    Computes contrastive loss between adapter features and text instructions,
    plus distillation loss to preserve original image features.
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()
        self.logit_scale = self.clip_model.logit_scale.exp()

    def _transform(self, n_px: int):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, image_features_from_adapter: torch.Tensor, 
                instruction_tokens: torch.Tensor, 
                original_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate contrastive and distillation losses.
        
        Args:
            image_features_from_adapter: Features from the adapter
            instruction_tokens: Tokenized task instructions  
            original_image: Original uncompressed image
            
        Returns:
            Dictionary with 'contrastive_loss' and 'distillation_loss'
        """
        loss_dict = {}

        # Contrastive Loss
        image_features_norm = image_features_from_adapter / image_features_from_adapter.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(instruction_tokens.to(self.device))
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = self.logit_scale * image_features_norm @ text_features_norm.t()
        
        batch_size = image_features_from_adapter.size(0)
        labels = torch.arange(batch_size, device=self.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_image.t(), labels)
        loss_dict['contrastive_loss'] = (loss_i + loss_t) / 2

        # Distillation Loss
        with torch.no_grad():
            processed_original_image = self._transform(self.clip_model.visual.input_resolution)(original_image)
            original_image_features = self.clip_model.encode_image(processed_original_image)
        
        loss_dict['distillation_loss'] = F.mse_loss(image_features_from_adapter, original_image_features.detach())

        return loss_dict


class ClipClsloss(nn.Module):
    """CLIP classification loss with feature distillation."""
    
    def __init__(self, args, device, processor=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.args = args
        self.device = device
        self.clip_model, _ = clip_modify.load("ViT-L/14", device=device)
        self.clip_model.eval()
        self.zeroshot_weights = self.zeroshot_classifier()
        self.processing = self._transform(224)
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained("openvla/openvla-7b")
        
    def forward(self, output, labels, original_image=None, saved_token_count=None, exp_name=None):
        output, _ = self.clip_model.encode_image(output, start_layer=2) 
        logits = output.type(self.zeroshot_weights.dtype) @ self.zeroshot_weights 
        
        loss = {}
        loss['clipcls'] = self.ce(logits, labels)

        # Predicted features
        pred_feat = [output.clone()]

        # Original features
        image = self.processing(original_image)
        with torch.no_grad():
            output, _ = self.clip_model.encode_image(image, start_layer=0) 
            ori_feat = [output.detach().clone()]

        perc_loss = torch.stack([
            nn.functional.mse_loss(p, o, reduction='none') 
            for p, o in zip(pred_feat, ori_feat)
        ]).squeeze()

        loss['clip_distill'] = perc_loss.mean()
        return loss

    def zeroshot_classifier(self):
        """Create zero-shot classifier weights."""
        # Simplified class names (should be expanded for full ImageNet)
        classnames = ["tench", "goldfish", "great white shark", "tiger shark"]
        templates = ['a photo of a {}.', 'a rendering of a {}.', 'a cropped photo of the {}.']
        
        import os
        os.makedirs('ImageNet_CLIP_TextEmb', exist_ok=True)
        path = f'ImageNet_CLIP_TextEmb/ImageNet_text_CLIP_emb_ViT-L-14.pt'
        
        if os.path.isfile(path):
            zeroshot_weights = torch.load(path)
            return zeroshot_weights.cuda()
        else:
            with torch.no_grad():
                zeroshot_weights = []
                for classname in classnames:
                    texts = [template.format(classname) for template in templates]
                    texts = self.processor.tokenizer(texts)
                    class_embeddings = self.clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
                torch.save(zeroshot_weights, path)
            return zeroshot_weights

    def _transform(self, n_px: int):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]