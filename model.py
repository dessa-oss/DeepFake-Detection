import torchvision.models as models
import torch.nn as nn


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


def create_model(use_hidden_layer, dropout):
    model = models.resnet18(pretrained=True)

    # Uncomment to freeze pre-trained layers
    # for param in model.parameters():
    #     param.requires_grad = False

    in_features = model.fc.in_features
    print(f'Input feature dim: {in_features}')

    if use_hidden_layer:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, 2)
        )

    else:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )
    
    print(model)

    model = model.cuda()
    return model

