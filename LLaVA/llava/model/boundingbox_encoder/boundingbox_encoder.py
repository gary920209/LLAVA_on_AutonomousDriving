import re
import torch
import torch.nn as nn

class CLIPBoundingBoxEncoder(nn.Module):
    # bb_input_dim, output_dim, intermediate_channels=bb_output_dim, bb_projector_type=bb_projector_type
    def __init__(self, input_channels, output_dim, hidden_dim=512, patch_size=16, num_layers=12, num_heads=4, image_size=336, bb_projector_type="linear"):
        """
        Mimics the CLIP encoder architecture for bounding box features.

        Args:
            input_channels (int): Number of input channels (e.g., number of classes for bounding box maps).
            patch_size (int): Size of each patch (e.g., 16x16).
            output_dim (int): Output dimensionality of the encoder.
            num_layers (int): Number of Transformer encoder layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimensionality of the Transformer.
        """
        super(CLIPBoundingBoxEncoder, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.projector_type = bb_projector_type

        # Patch embedding: Linear projection of flattened patches
        self.patch_embedding = nn.Conv2d(
            input_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )  # Output: (B, hidden_dim, H/patch_size, W/patch_size)

        # Learnable positional embeddings
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, (image_size // patch_size) ** 2, hidden_dim)
        )  # Assuming input size is 224x224

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4),
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    
    def build_boundingbox_projector(self):

        if self.projector_type == 'linear':
            return nn.Linear(self.hidden_dim, self.output_dim)

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.hidden_dim, self.output_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.output_dim, self.output_dim))
            return nn.Sequential(*modules)

        raise ValueError(f'Unknown projector type: {self.projector_type}')


    def forward(self, x):
        """
        Forward pass for the bounding box encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, input_channels, H, W).

        Returns:
            torch.Tensor: Encoded features of shape (B, N_patches, output_dim).
        """
        # Convert to patches and embed
        patches = self.patch_embedding(x)  # Shape: (B, hidden_dim, H/patch_size, W/patch_size)
        B, C, H_patches, W_patches = patches.shape

        # Flatten spatial dimensions and add positional embeddings
        patches = patches.flatten(2).permute(0, 2, 1)  # Shape: (B, N_patches, hidden_dim)
        patches += self.positional_embeddings  # Add positional embeddings

        # Transformer requires input shape (N_patches, B, hidden_dim)
        patches = patches.permute(1, 0, 2)

        # Pass through Transformer
        encoded_features = self.transformer(patches)  # Shape: (N_patches, B, hidden_dim)

        # Revert to (B, N_patches, hidden_dim) and project to output_dim
        encoded_features = encoded_features.permute(1, 0, 2)  # Shape: (B, N_patches, hidden_dim)
        output_features = self.output_projection(encoded_features)  # Shape: (B, N_patches, output_dim)

        return output_features

# Example usage:
if __name__ == "__main__":
    bbox_encoder = CLIPBoundingBoxEncoder(input_channels=35, patch_size=16, output_dim=4096)
    input_tensor = torch.randn(8, 35, 336, 336)  # Batch size 8, 35 classes, 224x224 resolution
    encoded_features = bbox_encoder(input_tensor)
    print(encoded_features.shape)  # Should be (8, 441, 4096) for 224x224 with patch size 16
