import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def GETcorrectnumber(loader, printcolor, num_classes):
    with torch.no_grad():
        self.num_classes = num_claseese
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(num_classes)]
        n_class_samples = [0 for i in range(num_classes)]
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = aoemnet(inputs)
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            # n_correct += (predicted == labels).sum().item()
            for k in range(predicted.shape[0]):
                if predicted[k]==labels[k]:
                    n_correct +=1
            # for i in range(num_classes): # accuracy for each class
            #     label = labels[i]
            #     pred = predicted[i]
            #     if (label == pred):
            #         n_class_correct[i] += 1
            #     n_class_samples[i] += 1
        acc = 100.0 * n_correct / n_samples
    print(printcolor+f'[{epoch + 1}] t accuracy： {acc}%'+printcolor)
    return acc

class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int):
                How long to wait after last time validation loss improved.
                Default: 7
            verbose (bool):
                If True, prints a message for each validation loss improvement.
                Default: False
            delta (float):
                Minimum change in the monitored quantity to qualify as an improvement.
                Default: 0
            path (str):
                Path for the checkpoint to be saved to.
                Default: 'checkpoint.pt'
            trace_func (function):
                trace print function.
                Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.compress = ChannelPool()
        self.spatialAttention = nn.Sequential(
            nn.Conv2d(2, 1, 7, 7, padding=3), #padding = (7-1)/2
            )

    def forward(self, x):
        # print('x',x.shape)
        x = self.compress(x)
        x = self.spatialAttention(x)
        # scale = F.sigmoid(x)
        scale = torch.sigmoid(x)
        # print('scale',scale.shape)
        
        return x * scale
    
class Flatten_MEG(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelAttention(nn.Module):
    """
            Implementation of a channel attention module.
        """
    class Showsize(nn.Module):
        def __init__(self):
            super(ChannelAttention.Showsize, self).__init__()
        def forward(self, x):
            # print(x.shape)
            return x

    def __init__(self, shape, reduction_factor=16):

        super(ChannelAttention, self).__init__()

        _, in_channel, h, w = shape
        self.mlp = nn.Sequential(
            # self.Showsize(),
            Flatten_MEG(),
            # self.Showsize(),
            nn.Linear(in_channel, in_channel // reduction_factor),
            nn.ReLU(),
            nn.Linear(in_channel // reduction_factor, in_channel),
        )
        # self.avg = nn.AvgPool2d(kernel_size=(h, w), stride=(h, w))
        # self.max = nn.MaxPool2d(kernel_size=(h, w), stride=(h, w))

    def forward(self, x):
        # print('x', x.shape)
        # avg = self.avg(x)
        # max = self.max(x)
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        sum = self.mlp(avg_pool) + self.mlp(max_pool)
        # print(sum.shape)
        scale = (
            torch.sigmoid(sum)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand_as(x)
        )

        return x * scale
    
class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, x, bp):

        # min_ = x.min(1, keepdim=True)[0]
        # if min_[0] < 0:
        #     x = x + min_
        # else:
        #     x = x - min_
        # x = x / x.max()
        x = x.view(x.shape[0], -1)
        bp = bp.view(bp.shape[0], -1)
        x = torch.cat([x, bp], -1)

        return x
    
class L1(torch.nn.Module):
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)