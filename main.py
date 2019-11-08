import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from apex import amp

from data_loader import create_dataloaders
from model import get_trainable_params, create_model, print_model_params
from train import train
from utils import parse_and_override_params

import foundations


# Fix random seed
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params = foundations.load_parameters()

data_dict = parse_and_override_params(params)

# Set job tags to easily spot data in use
foundations.set_tag(f'{data_dict[params["train_data"]]}: {params["train_data"]}')
# foundations.set_tag(f'big {params["train_data"]}')

print('Creating datasets')
# Get dataloaders
train_dl, val_base_dl, val_augment_dl, display_dl_iter = create_dataloaders(params)

print('Creating loss function')
# Loss function
criterion = nn.CrossEntropyLoss()

print('Creating model')
# Create model, freeze layers and change last layer
model = create_model(bool(params['use_hidden_layer']), params['dropout'])
_ = print_model_params(model)
params_to_update = get_trainable_params(model)

print('Creating optimizer')
# Create optimizer and learning rate schedules
optimizer = optim.Adam(params_to_update, lr=params['max_lr'], weight_decay=params['weight_decay'])
model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

# Learning rate scheme
if bool(params['use_lr_scheduler']):
    step_size_up = int(params['n_epochs'] * len(train_dl) * 0.3)
    step_size_down = params['n_epochs'] * len(train_dl) - step_size_up
    scheduler = lr_scheduler.OneCycleLR(optimizer, params['max_lr'], total_steps=None,
                                        epochs=params['n_epochs'], steps_per_epoch=len(train_dl),
                                        pct_start=params['pct_start'], anneal_strategy='cos',
                                        cycle_momentum=False)
else:
    scheduler = None

print('Training start..')
# Train
train(train_dl, val_base_dl, val_augment_dl, display_dl_iter, model, optimizer, params['n_epochs'], params['max_lr'], scheduler, criterion,
      train_source=params["train_data"])
