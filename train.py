import torch
from tqdm import tqdm
from apex import amp
import numpy as np

from utils import visualize_metrics, display_predictions_on_image
from sklearn.metrics import roc_auc_score as extra_metric

import foundations


class Records:
    def __init__(self):
        self.train_losses, self.train_losses_wo_dropout, self.base_val_losses, self.augment_val_losses = [], [], [], []
        self.train_accs, self.train_accs_wo_dropout, self.base_val_accs, self.augment_val_accs = [], [], [], []
        self.train_custom_metrics, self.train_custom_metrics_wo_dropout, self.base_val_custom_metrics, self.augment_val_custom_metrics = [], [], [], []
        self.lrs = []

    def write_to_records(self, **kwargs):
        assert len(set(kwargs.keys()) - set(self.__dir__())) == 0, 'invalid arguments!'
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def return_attributes(self):
        attributes = [i for i in self.__dir__() if not (i.startswith('__') and i.endswith('__') or i in ('write_to_records', 'return_attributes',
                                                                                                         'get_metrics'))]
        return attributes
    
    def get_metrics(self):
        return ['train_accs_wo_dropout', 'base_val_accs', 'augment_val_accs', 'base_val_custom_metrics', 'augment_val_custom_metrics']
    

def train_one_epoch(epoch, model, train_dl, max_lr, optimizer, criterion, extra_metric, scheduler, records):
    model.train()
    train_loss = 0
    train_loss_eval = 0
    train_tk = tqdm(train_dl, total=int(len(train_dl)), desc='Train Epoch')
    
    optimizer.zero_grad()
    total = 0
    correct_count = 0
    correct_count_eval = 0
    # f1_total = 0
    # f1_total_eval = 0
    
    for step, data in enumerate(train_tk):
        model.train()
        inputs = data['image']
        labels = data['label'].view(-1)
        
        inputs = inputs.cuda(device=0)
        labels = labels.cuda(device=0)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            
            correct_count += (predicted == labels).sum().item()
            # f1_total += extra_metric(labels.cpu().numpy(), predicted.cpu().numpy())
            
            loss = criterion(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            #             loss.backward()
            optimizer.step()
            if scheduler is not None:
                records.lrs += scheduler.get_lr()
                scheduler.step()
            else:
                records.lrs.append(max_lr)
        
        train_loss += loss.item()
        train_tk.set_postfix(loss=train_loss / (step + 1), acc=correct_count / total)
        
        # eval with dropout turned off
        model.eval()
        with torch.no_grad():
            outputs_eval = model(inputs)
            
            _, predicted_eval = torch.max(outputs_eval.data, 1)
            
            correct_count_eval += (predicted_eval == labels).sum().item()
            # f1_total_eval += extra_metric(labels.cpu().numpy(), predicted_eval.cpu().numpy())
            
            loss_eval = criterion(outputs_eval, labels)
        
        train_loss_eval += loss_eval.item()
    
    records.train_losses_wo_dropout.append(train_loss_eval / (step + 1))
    records.train_accs_wo_dropout.append(correct_count_eval / total)
    # records.train_custom_metrics_wo_dropout.append(f1_total_eval / len(train_dl))
    
    records.train_losses.append(train_loss / (step + 1))
    records.train_accs.append(correct_count / total)
    # records.train_custom_metrics.append(f1_total / len(train_dl))
    
    print(f'Epoch {epoch}: train loss={records.train_losses[-1]:.4f} | train acc={records.train_accs[-1]:.4f}')
    
    print(f'Epoch {epoch}: eval_ loss={records.train_losses_wo_dropout[-1]:.4f} | train acc={records.train_accs_wo_dropout[-1]:.4f}')


def validate(model, val_dl, criterion, extra_metric, records, data_name):
    # val
    model.eval()
    val_loss = 0
    correct_count = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    
    for data in val_dl:
        inputs = data['image']
        labels = data['label'].view(-1)
        
        inputs = inputs.cuda(device=0)  # .type()
        labels = labels.cuda(device=0)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct_count += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels)
            
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predicted.cpu().numpy())
        
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    extra_score = extra_metric(all_labels, all_predictions)
    
    if data_name == 'base':
        records.base_val_losses.append(val_loss / len(val_dl))
        records.base_val_accs.append(correct_count / total)
        records.base_val_custom_metrics.append(extra_score)
        print(f'\t base val loss={records.base_val_losses[-1]:.4f} | base val acc={records.base_val_accs[-1]:.4f} | '
              f'base val {extra_metric.__name__}={records.base_val_custom_metrics[-1]:.4f}')
        
    else:
        assert data_name == 'augment', f'specified data type is unknown {data_name}'
        records.augment_val_losses.append(val_loss / len(val_dl))
        records.augment_val_accs.append(correct_count / total)
        records.augment_val_custom_metrics.append(extra_score)
        print(f'\t augment val loss={records.augment_val_losses[-1]:.4f} |  augment val acc={records.augment_val_accs[-1]:.4f} | '
              f'augment val {extra_metric.__name__}={records.augment_val_custom_metrics[-1]:.4f}\n')


def train(train_dl, val_base_dl, val_augment_dl, display_dl_iter, model, optimizer, n_epochs, max_lr, scheduler, criterion, train_source):
    records = Records()
    
    for epoch in range(n_epochs):
        train_one_epoch(epoch, model, train_dl, max_lr, optimizer, criterion, extra_metric, scheduler, records)
        validate(model, val_base_dl, criterion, extra_metric, records, data_name='base')
        validate(model, val_augment_dl, criterion, extra_metric, records, data_name='augment')
        
        display_filename = f'{epoch}_display.png'
        display_predictions_on_image(model, val_base_dl.dataset.cached_path, display_dl_iter, name=display_filename)
        
        # Save eyeball plot to Atlas GUI
        foundations.save_artifact(display_filename, key=f'{epoch}_display')

        # Save metrics plot
        visualize_metrics(records, extra_metric=extra_metric, name='metrics.png')
        
        # Save metrics plot to Atlas GUI
        foundations.save_artifact('metrics.png', key='metrics_plot')
    
    # Log metrics to GUI
    if train_source == 'both':
        avg_metric = [getattr(records, 'base_val_accs'), getattr(records, 'augment_val_accs')]
        avg_metric = np.mean(avg_metric, axis=0)
        max_index = np.argmax(avg_metric)
    
    else:
        max_index = np.argmax(getattr(records, f'{train_source}_val_accs'))

    useful_metrics = records.get_metrics()
    for metric in useful_metrics:
        foundations.log_metric(metric, float(getattr(records, metric)[max_index]))
