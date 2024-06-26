from torch import nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
        
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(238144, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):

        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)
       
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out

