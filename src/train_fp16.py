import os

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import os
import sys
import time
import sklearn.metrics
import numpy as np


def cal_recall_score(outputs, targets):
    scores = []
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    for i,component in enumerate(['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']):
        y_true_subset = targets[i].cpu().numpy()
        y_pred_subset = outputs[i].cpu().numpy()
        scores.append(sklearn.metrics.recall_score(
            y_true_subset, y_pred_subset, average='macro'))
    final_score = np.average(scores, weights=[2,1,1])
    return final_score

def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)

    return (l1+l2+l3)/3

class Trainer(object):
    def __init__(self,
                 model_name,
                 model,
                 train_on_gpu=False,
                 fp16=False,
                 loss_scaling=False):
        self.model = model
        self.model_name = model_name
        self.train_on_gpu = train_on_gpu
        self.loss_scaling = loss_scaling
        if train_on_gpu and torch.backends.cudnn.enabled:
            self.fp16_mode = fp16
        else:
            self.fp16_mode = False
            self.loss_scaling = False
            print("CuDNN backend not available. Can't train with FP16.")

        self.best_acc = 0
        self.best_epoch = 0
        self._LOSS_SCALE = 128.0

        if self.train_on_gpu:
            self.model = self.model.cuda()

        if self.fp16_mode:
            self.model = self.network_to_half(self.model)
            self.model_params, self.master_params = self.prep_param_list(
                self.model)

        if self.train_on_gpu:
            self.model = nn.DataParallel(self.model)

        print('\n Model: {} | Training on GPU: {} | Mixed Precision: {} |'
              'Loss Scaling: {}'.format(self.model_name, self.train_on_gpu,
                                        self.fp16_mode, self.loss_scaling))

    def prep_param_list(self, model):
        """
        Create two set of of parameters. One in FP32 and other in FP16.
        Since gradient updates are with numbers that are out of range
        for FP16 this a necessity. We'll update the weights with FP32
        and convert them back to FP16.
        """
        model_params = [p for p in model.parameters() if p.requires_grad]
        master_params = [p.detach().clone().float() for p in model_params]

        for p in master_params:
            p.requires_grad = True

        return model_params, master_params

    def master_params_to_model_params(self, model_params, master_params):
        """
        Move FP32 master params to FP16 model params.
        """
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)

    def model_grads_to_master_grads(self, model_params, master_params):
        for model, master in zip(model_params, master_params):
            if master.grad is None:
                master.grad = Variable(master.data.new(*master.data.size()))
            master.grad.data.copy_(model.grad.data)

    def BN_convert_float(self, module):
        '''
        Designed to work with network_to_half.
        BatchNorm layers need parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.float()
        for child in module.children():
            self.BN_convert_float(child)
        return module

    class tofp16(nn.Module):
        """
        Add a layer so inputs get converted to FP16.
        Model wrapper that implements::
            def forward(self, input):
                return input.half()
        """

        def __init__(self):
            super(Trainer.tofp16, self).__init__()

        def forward(self, input):
            return input.half()

    def network_to_half(self, network):
        """
        Convert model to half precision in a batchnorm-safe way.
        """
        return nn.Sequential(self.tofp16(),
                             self.BN_convert_float(network.half()))

    def warmup_learning_rate(self, init_lr, no_of_steps, epoch, len_epoch):
        """Warmup learning rate for 5 epoch"""
        factor = no_of_steps // 30
        lr = init_lr * (0.1**factor)
        """Warmup"""
        lr = lr * float(1 + epoch + no_of_steps * len_epoch) / (5. * len_epoch)
        return lr

    def train(self, epoch, no_of_steps, trainloader, lr):
        self.model.train()

        train_loss, correct, total = 0, 0, 0

        # Declare optimizer.
        if not hasattr(self, 'optimizer'):
            if self.fp16_mode:
                self.optimizer = optim.Adam(
                    self.master_params, lr)
            else:
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr,)

        # If epoch less than 5 use warmup, else use scheduler.
        if epoch < 5:
            lr = self.warmup_learning_rate(lr, no_of_steps, epoch,
                                           len(trainloader))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == 5:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( self.optimizer, mode="min", 
                                                            factor = 0.3,
                                                            patience = 5,
                                                            verbose = False)
        if epoch >= 5:
            scheduler.step(epoch=epoch)

        print('Learning Rate: %g' % (list(
            map(lambda group: group['lr'], self.optimizer.param_groups)))[0])
        # Loss criterion is in FP32.

        for idx, d in tqdm(enumerate(trainloader), total = int(TRAIN_SIZE/trainloader.batch_size)):
            inputs = d["image"]
            grapheme_root = d["grapheme_root"]
            vowel_diacritic = d["vowel_diacritic"]
            consonant_diacritic = d["consonant_diacritic"]

            if self.train_on_gpu:
                inputs, grapheme_root,vowel_diacritic,consonant_diacritic = inputs.cuda().half(), grapheme_root.cuda(), vowel_diacritic.cuda(),consonant_diacritic.cuda()

            self.model.zero_grad()
            outputs = self.model(inputs)
            # We calculate the loss in FP32 since reduction ops can be
            # wrong when represented in FP16.
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            
            if self.loss_scaling:
                # Sometime the loss may become small to be represente in FP16
                # So we scale the losses by a large power of 2, 2**7 here.
                loss = loss * self._LOSS_SCALE
            # Calculate the gradients
            loss.backward()
            if self.fp16_mode:
                # Now we move the calculated gradients to the master params
                # so that we can apply the gradient update in FP32.
                self.model_grads_to_master_grads(self.model_params,
                                                 self.master_params)
                if self.loss_scaling:
                    # If we scaled our losses now is a good time to scale it
                    # back since our gradients are in FP32.
                    for params in self.master_params:
                        params.grad.data = params.grad.data / self._LOSS_SCALE
                # Apply weight update in FP32.
                self.optimizer.step()
                # Copy the updated weights back FP16 model weights.
                self.master_params_to_model_params(self.model_params,
                                                   self.master_params)
            else:
                self.optimizer.step()

    def evaluate(self, epoch, testloader):
        self.model.eval()

        test_loss = 0
        recall_score = 0

        criterion = loss_fn

        with torch.no_grad():
            for idx, d in enumerate(testloader):
                image = d["image"]
                grapheme_root = d["grapheme_root"]
                vowel_diacritic = d["vowel_diacritic"]
                consonant_diacritic = d["consonant_diacritic"]
                if self.train_on_gpu:
                    image, grapheme_root,vowel_diacritic,consonant_diacritic = image.cuda().half(), grapheme_root.cuda(),vowel_diacritic.cuda(),consonant_diacritic.cuda()
                outputs = self.model(image)
                targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                recall_score += cal_recall_score(outputs, targets)

        acc = recall_score/testloader.batch_size
        if acc > self.best_acc:
            self.save_model(self.model, self.model_name, acc, epoch)
            self.best_acc = recall_score

    def save_model(self, model, model_name, acc, epoch):
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if self.fp16_mode:
            save_name = os.path.join('weights', model_name + '_fp16',
                                     'weights.%03d.%.03f.pt' % (epoch, acc))
        else:
            save_name = os.path.join('weights', model_name,
                                     'weights.%03d.%.03f.pt' % (epoch, acc))

        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))

        torch.save(state, save_name)
        print("\nSaved state at %.03f%% accuracy. Prev accuracy: %.03f%%" %
              (acc, self.best_acc))
        self.best_acc = acc
        self.best_epoch = epoch

    def load_model(self, path=None):
        """
        Load previously saved model. THis doesn't check for precesion type.
        """
        if path is not None:
            checkpoint_name = path
        elif self.fp16_mode:
            checkpoint_name = os.path.join(
                'weights', self.model_name + '_fp16',
                'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        else:
            checkpoint_name = os.path.join(
                'weights', self.model_name + '_fp16',
                'weights.%03d.%.03f.pt' % (self.best_epoch, self.best_acc))
        if not os.path.exists(checkpoint_name):
            print("Best model not found")
            return
        checkpoint = torch.load(checkpoint_name)
        self.model.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.best_epoch = checkpoint['epoch']
        print("Loaded Model with accuracy: %.3f%%, from epoch: %d" %
              (checkpoint['acc'], checkpoint['epoch'] + 1))

    def train_and_evaluate(self, traindataloader, testdataloader, no_of_steps,
                           lr):
        self.best_acc = 0.0
        for i in range(no_of_steps):
            print('\nEpoch: %d' % (i + 1))
            self.train(i, no_of_steps, traindataloader, lr)
            self.evaluate(i, testdataloader)

import torch
import torch.nn as nn
import os
import ast
from modeldispatcher import MODELDISPATCHER
from dataset import BengaliDatasetTrain
from tqdm import tqdm


DEVICE = "cuda"
IMG_HEIGHT=137
IMG_WIDTH=236
EPOCHS=15
TRAIN_BATCH_SIZE=64
TEST_BATCH_SIZE=64
MODEL_MEAN=(0.485, 0.456, 0.406)
MODEL_STD=(0.229, 0.224, 0.225)
BASE_MODEL="efficientnet"
IMAGE_SIZE=128

TRAINING_FOLDS=[0,1,2,3]
VALIDATION_FOLDS=[4]


if __name__ == "__main__":
    if BASE_MODEL == "resnet34":
        model = MODELDISPATCHER[BASE_MODEL](pretrained = True)
    else :
        model = MODELDISPATCHER[BASE_MODEL]

    train_dataset = BengaliDatasetTrain(folds = TRAINING_FOLDS, 
                                        img_height= IMG_HEIGHT,
                                        img_width= IMG_WIDTH,
                                        mean = MODEL_MEAN,
                                        std = MODEL_STD,
                                        Image_size= IMAGE_SIZE)

    train_loader = torch.utils.data.DataLoader( train_dataset,
                                                batch_size= TRAIN_BATCH_SIZE,
                                                shuffle = True,
                                                num_workers = 0,)
    
    valid_dataset = BengaliDatasetTrain(folds = VALIDATION_FOLDS, 
                                        img_height= IMG_HEIGHT,
                                        img_width= IMG_WIDTH,
                                        mean = MODEL_MEAN,
                                        std = MODEL_STD,Image_size= IMAGE_SIZE)

    valid_loader = torch.utils.data.DataLoader( valid_dataset,
                                                batch_size= TEST_BATCH_SIZE,
                                                shuffle = False,
                                                num_workers = 0)
    TRAIN_SIZE = len(train_dataset)
    trainer = Trainer(BASE_MODEL, model,True, True, True )

    trainer.train_and_evaluate(train_loader,valid_loader,2, 1e-4 )

    