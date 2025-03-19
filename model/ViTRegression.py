class ViTRegression(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, img_size=256)


        # Modify input to accept single-channel (grayscale) images
        # ViT uses a Conv2d patch embedding layer (first layer)
        conv_proj = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv2d(
            1,
            conv_proj.out_channels,
            kernel_size=conv_proj.kernel_size,
            stride=conv_proj.stride,
            padding=conv_proj.padding,
            bias=False
        )

        # Modify final regression head
        num_features = self.model.head.in_features
        self.model.head = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)
