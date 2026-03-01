import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        out = x * self.channel_attention(x)
        # Apply spatial attention
        out = out * self.spatial_attention(out)
        return out


class AttentionCNN(nn.Module):
    """Attention-based CNN for Image Classification"""
    def __init__(self, num_classes, input_channels=3, dropout_rate=0.5, 
                 attention_reduction=16, spatial_kernel_size=7, fc_hidden_dim=256):
        super(AttentionCNN, self).__init__()
        
        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Attention block 1
        self.attention1 = CBAM(64, reduction=attention_reduction, kernel_size=spatial_kernel_size)
        
        # Convolution block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.attention2 = CBAM(128, reduction=attention_reduction, kernel_size=spatial_kernel_size)
        
        # Convolution block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.attention3 = CBAM(256, reduction=attention_reduction, kernel_size=spatial_kernel_size)
        
        # Convolution block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.attention4 = CBAM(512, reduction=attention_reduction, kernel_size=spatial_kernel_size)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.attention1(x)
        
        x = self.conv2(x)
        x = self.attention2(x)
        
        x = self.conv3(x)
        x = self.attention3(x)
        
        x = self.conv4(x)
        x = self.attention4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def create_model(num_classes, input_channels=3, pretrained=False, 
                 dropout_rate=0.5, attention_reduction=16, 
                 spatial_kernel_size=7, fc_hidden_dim=256):
    """
    Create an attention-based CNN model
    
    Args:
        num_classes: Number of classes to classify
        input_channels: Number of input channels (3 for RGB, 1 for grayscale)
        pretrained: Whether to use pretrained weights (not implemented yet)
        dropout_rate: Dropout rate for classifier (default: 0.5)
        attention_reduction: Reduction ratio for channel attention (default: 16)
        spatial_kernel_size: Kernel size for spatial attention (default: 7)
        fc_hidden_dim: Hidden dimension for fully connected layer (default: 256)
    
    Returns:
        AttentionCNN model
    """
    model = AttentionCNN(num_classes, input_channels, 
                         dropout_rate=dropout_rate,
                         attention_reduction=attention_reduction,
                         spatial_kernel_size=spatial_kernel_size,
                         fc_hidden_dim=fc_hidden_dim)
    return model