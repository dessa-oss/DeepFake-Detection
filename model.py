import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F


class SimpleCNN(torch.nn.Module):
    
    # Our batch shape for input x is (3, 32, 32)
    
    def __init__(self, hparams):
        super(SimpleCNN, self).__init__()
        
        self.hidden_channels = 64

        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, self.hidden_channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=5, stride=1, padding=2)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.hidden_channels, self.hidden_channels)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(self.hidden_channels, 2)
        
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
        self.drop4 = nn.Dropout(0.5)
        
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        self.bn2 = nn.BatchNorm2d(self.hidden_channels)
        self.bn3 = nn.BatchNorm2d(self.hidden_channels)
        self.bn4 = nn.BatchNorm2d(self.hidden_channels)
        
        self.dense_drop = nn.Dropout(0.5)
    
    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.drop1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.drop2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.drop3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.drop4(x)

        # Size changes from (18, 32, 32) to (18, 16, 16)
        # x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        # x = x.view(-1, 18 * 16 * 16)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.squeeze(x)
    
        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        x = self.dense_drop(x)

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return x
    

def check_model_block(model):
    for name, child in model.named_children():
        print(name)


def print_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of params: {pytorch_total_params:,}')
    return pytorch_total_params
    

def get_trainable_params(model):
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", repr(name))
            params_to_update.append(param)
            
    return params_to_update

#
# def create_model(hparams):
#     model = SimpleCNN(hparams).cuda()
#     return model


def create_model(dropout):
    model = models.resnet18(pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False

    in_features = model.fc.in_features
    print(f'Input feature dim: {in_features}')

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, in_features // 2),
        nn.ReLU(),
        nn.BatchNorm1d(in_features // 2),
        nn.Dropout(dropout),
        nn.Linear(in_features // 2, 2)
    )
    
    print(model)

    model = model.cuda()
    return model

