
import torch
import torch.nn as nn

class PatchingLayer(nn.Module):
    """
    Transforms a time series into a sequence of patches and projects them
    into the model dimension (d_model).
    """
    def __init__(self, patch_len: int, stride: int, d_model: int, input_features: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.input_features = input_features

        # Linear projection for each patch
        self.projection = nn.Linear(self.patch_len * self.input_features, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input time series of shape [Batch, SeqLen, Features].

        Returns:
            torch.Tensor: A sequence of patch embeddings of shape [Batch, NumPatches, d_model].
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Ensure sequence length is compatible with patching
        if (seq_len - self.patch_len) % self.stride != 0:
            # Simple padding on the right to make it fit
            padding_len = self.stride - ((seq_len - self.patch_len) % self.stride)
            x = nn.functional.pad(x, (0, 0, 0, padding_len), mode='replicate')
            seq_len = x.shape[1]

        # Unfold the sequence into patches
        # Input: [Batch, SeqLen, Features] -> [Batch, Features, SeqLen]
        x = x.permute(0, 2, 1)
        # Unfold: -> [Batch, Features, NumPatches, PatchLen]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # Reshape for projection
        # [Batch, Features, NumPatches, PatchLen] -> [Batch, NumPatches, Features, PatchLen]
        patches = patches.permute(0, 2, 1, 3)
        # [Batch, NumPatches, Features, PatchLen] -> [Batch, NumPatches, Features * PatchLen]
        patches = patches.reshape(batch_size, -1, self.input_features * self.patch_len)
        
        # Project patches into d_model
        patch_embeddings = self.projection(patches)
        
        return patch_embeddings
