"""
FastVLM-7B Remote Zoo Model for FiftyOne
Implements Apple's FastVLM-7B vision-language model as a FiftyOne zoo model
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.core.labels import Classification, Classifications
from fiftyone.operators import types
from fiftyone.utils.torch import GetItem

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "apple/FastVLM-7B"
IMAGE_TOKEN_INDEX = -200  # Special token for image placeholder

DEFAULT_VQA_SYSTEM_PROMPT = """You are a helpful assistant. You provide clear and concise answers to questions about images. Report answers in natural language text in English."""

# Operation modes and their default prompts
OPERATIONS = {
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
}


def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FastVLMGetItem(GetItem):
    """GetItem transform for loading images and sample data for FastVLM."""
    
    def __init__(self, field_mapping=None, prompt_field=None):
        """
        Initialize the GetItem transform.
        
        Args:
            field_mapping: Optional dict mapping required keys to dataset fields
            prompt_field: Name of the field containing prompts (optional)
        """
        # Set prompt_field BEFORE calling super().__init__()
        # because parent init accesses required_keys which needs prompt_field
        self.prompt_field = prompt_field
        super().__init__(field_mapping=field_mapping)
    
    @property
    def required_keys(self):
        """Return list of fields needed from each sample."""
        keys = ["filepath"]
        if self.prompt_field:
            keys.append(self.prompt_field)
        return keys
    
    def __call__(self, sample_dict):
        """
        Load image and extract prompt from sample.
        
        Args:
            sample_dict: Dict with keys from required_keys
            
        Returns:
            Dict with 'image' and 'prompt' keys
        """
        filepath = sample_dict["filepath"]
        image = Image.open(filepath).convert("RGB")
        
        # Extract prompt if available
        prompt = None
        if self.prompt_field and self.prompt_field in sample_dict:
            prompt = sample_dict[self.prompt_field]
        
        return {
            'image': image,
            'prompt': prompt,
            'sample_dict': sample_dict  # Keep for any other fields needed
        }


class FastVLM(Model, SupportsGetItem, TorchModelMixin):
    """
    A FiftyOne model for running Apple's FastVLM-7B vision-language model.
    
    This model performs Visual Question Answering (VQA) on images.
    """
    
    def __init__(
        self,
        model_path: str,
        prompt: str = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.90,
        top_k: int = 50,
        **kwargs
    ):
        """
        Initialize the FastVLM model.
        
        Args:
            model_path: Path to the downloaded model
            prompt: Default prompt/question for all images
            system_prompt: Custom system prompt (optional)
            temperature: Generation temperature (0.1-2.0)
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        # Initialize base classes
        SupportsGetItem.__init__(self)
        
        # Required for SupportsGetItem
        self._preprocess = False
        
        self._fields = {}
        
        self.model_path = model_path
        self.prompt = prompt or "What is in this image?"
        self._custom_system_prompt = system_prompt
        
        # Generation parameters
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        # Get device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Determine torch dtype based on device
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Enable bfloat16 on Ampere+ GPUs (compute capability 8.0+)
            if capability[0] >= 8:
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        # Load tokenizer
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        logger.info("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit - clear GPU memory cache."""
        # Clear cache based on device type (don't move model to CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False
    
    
    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields
    
    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        """Get the prompt field from needs_fields."""
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
        
        return prompt_field
    
    @property
    def media_type(self):
        """Returns the media type for the model."""
        return "image"
    
    @property
    def system_prompt(self):
        """Return custom system prompt if set, otherwise return default."""
        return self._custom_system_prompt if self._custom_system_prompt is not None else DEFAULT_VQA_SYSTEM_PROMPT
    
    @system_prompt.setter
    def system_prompt(self, value):
        """Set a custom system prompt."""
        self._custom_system_prompt = value
    
    # ============ Required Properties from Model ============
    
    @property
    def transforms(self):
        """Preprocessing transforms (None for SupportsGetItem models)."""
        return None
    
    @property
    def preprocess(self):
        """Whether model should apply preprocessing."""
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        """Allow FiftyOne to control preprocessing."""
        self._preprocess = value
    
    @property
    def ragged_batches(self):
        """Must return False to enable batching."""
        return False
    
    # ============ Required Properties from TorchModelMixin ============
    
    @property
    def has_collate_fn(self):
        """Whether this model provides custom batch collation."""
        return True
    
    @property
    def collate_fn(self):
        """Custom collation function for batching."""
        @staticmethod
        def identity_collate(batch):
            """Return batch as-is without stacking."""
            return batch
        return identity_collate
    
    # ============ Required Methods from SupportsGetItem ============
    
    def build_get_item(self, field_mapping=None):
        """Build the GetItem transform for data loading."""
        prompt_field = self._get_field()
        return FastVLMGetItem(field_mapping=field_mapping, prompt_field=prompt_field)
    
    # ============ Core Inference Methods ============
    
    def _predict(self, image: Image.Image, sample=None):
        """Process a single image through the model and return VQA response."""
        # Get prompt from sample field if available
        current_prompt = self.prompt
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                current_prompt = str(field_value)
        
        # Build the user content with system prompt and question
        user_content = f"{self.system_prompt}\n\nQuestion: {current_prompt}"
        
        # Build chat messages with <image> placeholder
        messages = [
            {"role": "user", "content": f"<image>\n{user_content}"}
        ]
        
        # Render to string (not tokens) so we can place <image> exactly
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Split at image token
        pre, post = rendered.split("<image>", 1)
        
        # Tokenize the text around the image token
        pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Splice in the IMAGE token id at the placeholder position
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)
        
        # Process image through model's vision tower
        px = self.model.get_vision_tower().image_processor(images=image, return_tensors="pt")["pixel_values"]
        px = px.to(self.model.device, dtype=self.torch_dtype)
        
        # Generate response
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Extract just the assistant's response if present
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        # Return as Classification with the response as the label
        return response.strip()
    
    def predict(self, image, sample=None):
        """
        Process an image with the model for Visual Question Answering.
        
        Args:
            image: PIL Image, numpy array, or path to an image
            sample: Optional FiftyOne sample containing the image
            
        Returns:
            fo.Classification containing the VQA response as the label
        """
        # Convert input to PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        return self._predict(image, sample)
    
    def predict_all(self, batch, preprocess=None):
        """
        Process a batch of samples with the model.
        
        This method attempts true batch inference without padding.
        Let's see what happens!
        
        Args:
            batch: List of dicts from GetItem with 'image' and 'prompt' keys
            preprocess: Whether to apply preprocessing (unused, handled by GetItem)
            
        Returns:
            List of responses (one per sample)
        """
        if not batch:
            return []
        
        # Extract images and prompts from batch
        images = []
        prompts = []
        
        for item in batch:
            images.append(item['image'])
            # Use per-sample prompt if available, otherwise use default
            prompt = item.get('prompt') or self.prompt
            prompts.append(prompt)
        
        # Try batch processing!
        try:
            # Build all the tokenized inputs
            all_input_ids = []
            all_attention_masks = []
            
            for prompt in prompts:
                # Build the user content with system prompt and question
                user_content = f"{self.system_prompt}\n\nQuestion: {prompt}"
                
                # Build chat messages with <image> placeholder
                messages = [
                    {"role": "user", "content": f"<image>\n{user_content}"}
                ]
                
                # Render to string
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                # Split at image token
                pre, post = rendered.split("<image>", 1)
                
                # Tokenize the text around the image token
                pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
                post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
                
                # Splice in the IMAGE token id at the placeholder position
                img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
                input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
                
                all_input_ids.append(input_ids)
                all_attention_masks.append(torch.ones_like(input_ids))
            
            # Try to stack/concatenate the inputs (this might fail if sequences have different lengths)
            # Let's see what happens when we just concatenate along batch dimension
            batched_input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
            batched_attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)
            
            # Process all images through vision tower at once
            # The image_processor should handle batching
            px = self.model.get_vision_tower().image_processor(images=images, return_tensors="pt")["pixel_values"]
            px = px.to(self.model.device, dtype=self.torch_dtype)
            
            # Generate response for the batch
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs=batched_input_ids,
                    attention_mask=batched_attention_mask,
                    images=px,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode all responses
            results = []
            for i in range(len(batch)):
                response = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                
                # Extract just the assistant's response if present
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                
                results.append(response.strip())
            
            return results
            
        except Exception as e:
            # If batch processing fails, fall back to one-by-one processing
            logger.warning(f"Batch inference failed ({str(e)}), falling back to sequential processing")
            results = []
            for item in batch:
                response = self._predict(item['image'], None)
                results.append(response)
            return results