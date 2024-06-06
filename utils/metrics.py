import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self, average='micro', num_classes=1, smooth = 1e-6):
        super(Dice, self).__init__()
        self.average = average
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, input, target):
        # input and target are of shape (batch_size, num_classes, H, W)
        input = input.view(input.size(0), self.num_classes, -1)
        target = target.view(target.size(0), self.num_classes, -1)  
        intersection = (input * target).sum(2)
        if self.average == 'micro':
            return ((2. * intersection + self.smooth) / (input.sum(2) + target.sum(2) + self.smooth)).mean()
        else:
            raise NotImplementedError("Only micro average is implemented")
    
        

class DiceLoss(nn.Module):
    def __init__(self, average='micro', num_classes=1):
        super(DiceLoss, self).__init__()
        self.average = average
        self.num_classes = num_classes

    def forward(self, input, target):
        # input and target are of shape (batch_size, num_classes, H, W)
        smooth = 1e-6
        input = input.view(input.size(0), self.num_classes, -1)
        target = target.view(target.size(0), self.num_classes, -1)
        intersection = (input * target).sum(2)
        if self.average == 'micro':
            return 1 - ((2. * intersection + smooth) / (input.sum(2) + target.sum(2) + smooth)).mean()
        else:
            raise NotImplementedError("Only micro average is implemented")
        


class DiceBceLoss(nn.Module):
    def __init__(self, average='micro', num_classes=1, weight_dice=1, weight_bce=1):
        super(DiceBceLoss, self).__init__()
        self.average = average
        self.num_classes = num_classes
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.dice = Dice(average=average, num_classes=num_classes)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.weight_dice * self.dice(input, target) + self.weight_bce * self.bce(input, target)
        

