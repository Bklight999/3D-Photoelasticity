import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os

n_channels=3

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode='reflect')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            self.pad,
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            self.pad,
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Define UNet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 1
        self.down4 = (Down(512, 1024 // factor))
        
        self.up1 = (Up(1024, 512 // factor))
        self.up2 = (Up(512, 256 // factor))
        self.up3 = (Up(256, 128 // factor))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Custom dataset
class ImageToImageDataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_images[idx])
        target_image = Image.open(self.target_images[idx])
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

# Define data paths
input_image_folder = "/mnt/sda/qifan/UNET/My_own_UNET/Fringe_colors"
target_image_folder = "/mnt/sda/qifan/UNET/My_own_UNET/Stress_maps"

input_image_paths = [os.path.join(input_image_folder, filename) for filename in os.listdir(input_image_folder) if filename.endswith('.bmp') or filename.endswith('.png')]
target_image_paths = [os.path.join(target_image_folder, filename) for filename in os.listdir(target_image_folder) if filename.endswith('.bmp') or filename.endswith('.png')]

batch_size = 32
learning_rate = 0.0001 
num_epochs = 10

# # Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),               # 转换为张量
])

# Define dataset
dataset = ImageToImageDataset(input_image_paths, target_image_paths, transform=transform)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Load train and test dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for better training
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        outputs = model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item()}")

# Test model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        test_loss += criterion(outputs, targets)

print(f"Test Loss: {test_loss / len(test_dataloader)}")

# Save model
torch.save(model.state_dict(), "unet_model_reflect.pth")
