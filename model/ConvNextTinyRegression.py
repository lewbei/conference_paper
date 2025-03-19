class ConvNextTinyRegression(nn.Module):
    def __init__(self, pretrained=False):
        super(ConvNextTinyRegression, self).__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=pretrained)

        # Modify first convolution to handle single-channel images
        first_conv = self.model.stem[0]
        self.model.stem[0] = nn.Conv2d(
            1, 
            first_conv.out_channels, 
            kernel_size=first_conv.kernel_size, 
            stride=first_conv.stride, 
            padding=first_conv.padding, 
            bias=(first_conv.bias is not None)

        )

        # Replace classifier for single output regression
        num_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)
