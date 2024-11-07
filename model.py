import torch

class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)

        self.flattened_size = self._get_flattened_size()
        
        self.fc1 = torch.nn.Linear(self.flattened_size, 128)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 2)

    def _get_flattened_size(self):
        x = torch.randn(1, 3, 50, 50)
        x = self.pool(torch.nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(torch.nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.pool(torch.nn.ReLU()(self.bn4(self.conv4(x))))
        return x.view(-1).size(0)

    def forward(self, x):
        x = self.pool(torch.nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(torch.nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.pool(torch.nn.ReLU()(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.nn.ReLU()(self.fc1(x)))
        x = self.fc2(x)
        return x