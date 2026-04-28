from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from monai.networks.nets.swin_unetr import SwinTransformer
import torch
import torch.nn as nn

from .dino_r50 import model


class EchoCareSwinTransformer(nn.Module):
    def __init__(self, pretrained_checkpoint=None):
        super().__init__()

        self.encoder = SwinTransformer(
            in_chans=3,
            embed_dim=128,
            window_size=[8] * 2,
            patch_size=[2] * 2,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            use_checkpoint=False,
            spatial_dims=2,
            use_v2=True)

        if pretrained_checkpoint is not None:
            self._load_pretrained_weights(pretrained_checkpoint)

    def _load_pretrained_weights(self, pretrained_checkpoint):
        model_dict = torch.load(pretrained_checkpoint, map_location=torch.device('cpu'))
        state_dict = model_dict
        state_dict.pop('mask_token')
        self.encoder.load_state_dict(state_dict, strict=True)
        print("Using pretrained self-supervised Swin Transformer backbone weights !")

    def forward(self, x):
        x_list = self.encoder(x)
        return {
            "p1": x_list[2],
            "p2": x_list[3],
            "p3": x_list[4],
        }

# modify backbone config
model.backbone = L(EchoCareSwinTransformer)()

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=512),
    "p2": ShapeSpec(channels=1024),
    "p3": ShapeSpec(channels=2048),
}
model.neck.in_features = ["p1", "p2", "p3"]
