import torch
from torch import nn
import torch.nn.functional as F
import deepspeed

class TransformerBlock(nn.Module):
    """Base class for transformer blocks with memory management"""
    def __init__(self, block):
        super().__init__()
        self.block = block
        
    @torch.cuda.amp.autocast()
    def forward(self, inputs):
        try:
            output = self._forward_impl(inputs)
            return output
        finally:
            # Cleanup intermediate tensors
            torch.cuda.empty_cache()
            
    def _forward_impl(self, inputs):
        raise NotImplementedError


class DoubleBlock(TransformerBlock):
    def _forward_impl(self, inputs):
        latents, t, text_embeds, text_embeds_2 = inputs
        output = self.block(
            latents, 
            t,
            text_states=text_embeds,
            text_states_2=text_embeds_2
        )
        return output, t, text_embeds, text_embeds_2


class SingleBlock(TransformerBlock):
    def _forward_impl(self, inputs):
        latents, t, text_embeds, text_embeds_2 = inputs
        output = self.block(
            latents,
            t, 
            text_states=text_embeds,
            text_states_2=text_embeds_2
        )
        return output, t, text_embeds, text_embeds_2


class OutputLayer(TransformerBlock):
    def _forward_impl(self, inputs):
        latents, t, text_embeds, text_embeds_2 = inputs
        return self.block.final_layer(latents)


class CheckpointWrapper(nn.Module):
    """Wrapper for activation checkpointing"""
    def __init__(self, module, checkpoint_func):
        super().__init__()
        self.module = module
        self.checkpoint_func = checkpoint_func
        
    def forward(self, *args, **kwargs):
        return self.checkpoint_func(self.module, *args, **kwargs)


class HunyuanVideoWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        try:
            self.config = config
            self.model_config = config['model']
            
            # Initialize components
            self._initialize_components()
            
        except Exception as e:
            print(f"Error initializing HunyuanVideoWrapper: {str(e)}")
            raise

    def _initialize_components(self):
        """Initialize all model components with proper device placement"""
        try:
            # Load transformer with empty weights first
            transformer_dtype = self.model_config.get('transformer_dtype', self.model_config['dtype'])
            factor_kwargs = {"device": 'cuda', "dtype": transformer_dtype}
            in_channels = 16  # latent channels
            out_channels = 16
            
            with init_empty_weights():
                self.transformer = load_model(
                    self.config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    factor_kwargs=factor_kwargs,
                )
            
            # Load transformer weights
            if transformer_path := self.model_config.get('transformer_path', None):
                state_dict = load_safetensors(transformer_path)
                state_dict = _convert_state_dict_keys(self.transformer.state_dict(), state_dict)
            else:
                state_dict = load_state_dict(self.config, self.config.get('model_base'))
                
            # Set parameters with proper dtypes
            params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
            base_dtype = self.model_config['dtype']
            for name, param in self.transformer.named_parameters():
                dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else transformer_dtype
                set_module_tensor_to_device(self.transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
                param.original_name = name  # Store original name for saving
            
            # Load VAE and text encoders
            self.vae = self._load_vae()
            self.text_encoder, self.text_encoder_2 = self._load_text_encoders()
            
            # Set proper devices
            self.vae.to('cpu').eval().requires_grad_(False)
            self.text_encoder.to('cpu')
            self.text_encoder_2.to('cpu')
            self.transformer.train()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def to_layers(self):
        """Convert model into pipeline-parallel layers"""
        layers = []
        
        # Initial layer handles VAE encoding and input preparation
        layers.append(InitialLayer(self.vae))
        
        # Add transformer blocks with proper checkpointing
        if self.config.get('activation_checkpointing', False):
            checkpoint_func = deepspeed.checkpointing.checkpoint
            for block in self.transformer.double_blocks:
                layers.append(
                    CheckpointWrapper(
                        DoubleBlock(block),
                        checkpoint_func
                    )
                )
            for block in self.transformer.single_blocks:
                layers.append(
                    CheckpointWrapper(
                        SingleBlock(block),
                        checkpoint_func
                    )
                )
        else:
            for block in self.transformer.double_blocks:
                layers.append(DoubleBlock(block))
            for block in self.transformer.single_blocks:
                layers.append(SingleBlock(block))
        
        # Final output layer
        layers.append(OutputLayer(self.transformer))
        
        return layers 