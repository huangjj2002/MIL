

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EDLHead(nn.Module):
    """
        evidence = Softplus(Linear(x))   shape: (B, K)
        alpha = evidence + 1              
        S = sum(alpha)                    
        p_k = alpha_k / S              
        uncertainty = K / S               
    """
    def __init__(self, in_features, num_classes=2, dropout=0.0):
        super(EDLHead, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)
        self.softplus = nn.Softplus()
    
    def forward(self, x):

        x = self.drop(x)
        evidence = self.softplus(self.linear(x))  # (B, K), ensure evidence >= 0
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)  # (B, 1)
        prob = alpha / S  # (B, K)
        uncertainty = alpha.shape[-1] / S.squeeze(-1)  # (B,)
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'S': S,
            'prob': prob,
            'uncertainty': uncertainty,
        }


class MIL_EDL_Wrapper(nn.Module):

    def __init__(self, mil_model, edl_dropout=0.0):
        super(MIL_EDL_Wrapper, self).__init__()
        
        self.mil_model = mil_model
        self.is_training = True
        

        self.mil_type = getattr(mil_model, 'mil_type', 'embedding')
        self.multi_scale_model = mil_model.multi_scale_model
        self.scales = getattr(mil_model, 'scales', None)
        self.type_scale_aggregator = getattr(mil_model, 'type_scale_aggregator', None)
        self.deep_supervision = getattr(mil_model, 'deep_supervision', False)
        self.pooling_type = mil_model.pooling_type
        
  
        if hasattr(mil_model, 'classifier'):
            in_features = self._get_classifier_in_features(mil_model.classifier)
            num_classes = mil_model.num_classes
     
            self.edl_head = EDLHead(in_features, num_classes=2, dropout=edl_dropout)
      
        self.edl_side_heads = nn.ModuleDict()
        if hasattr(mil_model, 'side_classifiers'):
            for key, clf in mil_model.side_classifiers.items():
                in_feat = self._get_classifier_in_features(clf)
                self.edl_side_heads[key] = EDLHead(in_feat, num_classes=2, dropout=edl_dropout)
    
    def _get_classifier_in_features(self, classifier_module):
  
        if isinstance(classifier_module.head_classifier, nn.Sequential):
            linear = classifier_module.head_classifier[0]
        else:
            linear = classifier_module.head_classifier
        return linear.in_features
    
    def _run_encoder_aggregator(self, x, bag_mask=None):

        model = self.mil_model
        
  
        if model.inst_encoder is not None:
            if isinstance(x, list):
                batch_size, num_patches, _, _, _ = x[0].size()
                x = [tensor.view(-1, tensor.size(2), tensor.size(3), tensor.size(4)) for tensor in x]
            else:
                batch_size, num_patches, C, H, W = x.size()
                x = x.view(-1, C, H, W)
            x = model.inst_encoder(x)
            
            if model.multi_scale_model in ['fpn', 'backbone_pyramid']:
                from collections import OrderedDict
                x_pyramid = OrderedDict()
                for key, fmap in x.items():
                    _, channels, height, width = fmap.shape
                    fmap = fmap.view(batch_size, num_patches, channels, height, width)
                    fmap = fmap.permute(0, 1, 3, 4, 2)
                    fmap = fmap.reshape(fmap.size(0), -1, fmap.size(4))
                    x_pyramid[key] = fmap
                x = x_pyramid
            elif model.multi_scale_model == 'msp':
                from collections import OrderedDict
                x_pyramid = OrderedDict()
                for idx, scale in enumerate(self.scales):
                    x_scale = x[scale] if isinstance(x, dict) else x
                    if not isinstance(x, dict):
                        batch_size_s, num_patches_s, C_s, H_s, W_s = x_scale.size()
                        x_scale = x_scale.view(-1, C_s, H_s, W_s)
                        x_scale = model.inst_encoder(x_scale)
                        x_pyramid[f'feat_{idx}'] = x_scale.view(batch_size, num_patches_s, -1)
                    else:
                        x_pyramid[f'feat_{idx}'] = x_scale
                x = x_pyramid
        
        return x
    
    def forward(self, x, bag_mask=None):

        model = self.mil_model
        
   
        if self.mil_type == 'embedding' and not self.multi_scale_model:
        
            if model.inst_encoder is not None:
                batch_size, num_patches, C, H, W = x.size()
                x_feat = x.view(-1, C, H, W)
                x_feat = model.inst_encoder(x_feat)
                x_feat = x_feat.view(batch_size, num_patches, -1)
            else:
                x_feat = x
        
            from MIL.AttentionModels import SetAttentionBlock, InducedSetAttentionBlock
            for block_encoder in model.encoder:
                if isinstance(block_encoder, (SetAttentionBlock, InducedSetAttentionBlock)):
                    x_feat = block_encoder(x_feat, bag_mask)
                else:
                    x_feat = block_encoder(x_feat)
      
            if self.pooling_type in ["attention", "gated-attention", "pma"]:
                bag_feat, A = model.aggregator(x_feat, bag_mask)
            else:
                bag_feat = model.aggregator(x_feat, bag_mask)
         
            edl_out = self.edl_head(bag_feat)
            edl_out['type'] = 'single_scale'
            return edl_out
        
  
        else:
            return self._forward_pyramidal(x, bag_mask)
    
    def _forward_pyramidal(self, x, bag_mask=None):
     
        from collections import OrderedDict
        from MIL.AttentionModels import SetAttentionBlock, InducedSetAttentionBlock
        model = self.mil_model
        
      
        x_pyramid = None
        if model.inst_encoder is not None:
            if isinstance(x, list):
                batch_size, num_patches, _, _, _ = x[0].size()
                x = [tensor.view(-1, tensor.size(2), tensor.size(3), tensor.size(4)) for tensor in x]
            else:
                batch_size, num_patches, C, H, W = x.size()
                x = x.view(-1, C, H, W)
            
            x_enc = model.inst_encoder(x)
            
            x_pyramid = OrderedDict()
            for key, fmap in x_enc.items():
                _, channels, height, width = fmap.shape
                fmap = fmap.view(batch_size, num_patches, channels, height, width)
                fmap = fmap.permute(0, 1, 3, 4, 2)
                fmap = fmap.reshape(fmap.size(0), -1, fmap.size(4))
                x_pyramid[key] = fmap
        else:
            if model.multi_scale_model == 'msp':
                x_pyramid = OrderedDict({
                    f'feat_{idx}': x[scale] for idx, scale in enumerate(self.scales)
                })
        
        scale_outputs = []
        side_edl_outputs = {}
        
        for scale in self.scales:
            x_patches = x_pyramid[f'feat_{self.scales.index(scale)}']
            
       
            for block_encoder in model.side_inst_aggregator['encoders'][f'encoder_{scale}']:
                if isinstance(block_encoder, (SetAttentionBlock, InducedSetAttentionBlock)):
                    x_patches = block_encoder(x_patches, bag_mask)
                else:
                    x_patches = block_encoder(x_patches)
            
            
            if self.pooling_type in ["attention", "gated-attention", "pma"]:
                x_patches, A = model.side_inst_aggregator['aggregators'][f'aggregator_{scale}'](x_patches, bag_mask)
            else:
                x_patches = model.side_inst_aggregator['aggregators'][f'aggregator_{scale}'](x_patches, bag_mask)
            
            scale_outputs.append(x_patches)
            
     
            if f'classifier_{scale}' in self.edl_side_heads:
                side_edl_out = self.edl_side_heads[f'classifier_{scale}'](x_patches)
                side_edl_outputs[scale] = side_edl_out
        
        if self.type_scale_aggregator in ['mean_p', 'max_p']:
            if not side_edl_outputs:
                raise RuntimeError(
                    f"EDL {self.type_scale_aggregator} aggregation requires scale-specific EDL heads."
                )

            evidences = torch.stack(
                [side_edl_outputs[scale]['evidence'] for scale in self.scales],
                dim=1
            )
            if self.type_scale_aggregator == 'mean_p':
                evidence = evidences.mean(dim=1)
            else:
                evidence = evidences.max(dim=1).values

            alpha = evidence + 1.0
            S = torch.sum(alpha, dim=-1, keepdim=True)
            prob = alpha / S
            uncertainty = alpha.shape[-1] / S.squeeze(-1)

            return {
                'evidence': evidence,
                'alpha': alpha,
                'S': S,
                'prob': prob,
                'uncertainty': uncertainty,
                'type': 'multi_scale',
                'side_outputs': side_edl_outputs,
            }

        x_stacked = torch.stack(scale_outputs, dim=1)  # (B, num_scales, feat_dim)
        
        if self.type_scale_aggregator == 'gated-attention':
            x_agg, A = model.scale_aggregator(x_stacked)
        elif self.type_scale_aggregator == 'concatenation':
            x_agg = model.scale_aggregator(x_stacked)
        else:
            raise ValueError(f"Unsupported scale aggregator for EDL: {self.type_scale_aggregator}")
        

        edl_out = self.edl_head(x_agg)
        edl_out['type'] = 'multi_scale'
        edl_out['side_outputs'] = side_edl_outputs
        
        return edl_out
