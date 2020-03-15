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
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=32
MODEL_MEAN=(0.485, 0.456, 0.406)
MODEL_STD=(0.229, 0.224, 0.225)
BASE_MODEL="efficientnet"

TRAINING_FOLDS=[0,1,2,3]
VALIDATION_FOLDS=[4]

def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)

    return (l1+l2+l3)/3


    

def train(dataset, dataloader, model, optimizer):
    model.train()
    for bi, d in tqdm(enumerate(dataloader), total = int(len(dataset)/dataloader.batch_size)):
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        image = image.to(DEVICE, dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype = torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


def evaluate(dataset, dataloader, model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(enumerate(dataloader), total = int(len(dataset)/dataloader.batch_size)):
        counter+=1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        image = image.to(DEVICE, dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype = torch.long)
        
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        final_loss += loss
    
    return final_loss / counter


def train_model(model,train_dataset,train_loader,valid_dataset,valid_loader,optimizer,scheduler ):
    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        with torch.no_grad():
            val_score = evaluate(valid_dataset,valid_loader, model)
        scheduler.step(val_score)

        torch.save(model.state_dict(), f"../saved_models/{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")

def main():
    if BASE_MODEL == "resnet34":
        model = MODELDISPATCHER[BASE_MODEL](pretrained = True)
    else :
        model = MODELDISPATCHER[BASE_MODEL]
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(folds = TRAINING_FOLDS, 
                                        img_height= IMG_HEIGHT,
                                        img_width= IMG_WIDTH,
                                        mean = MODEL_MEAN,
                                        std = MODEL_STD)

    train_loader = torch.utils.data.DataLoader( train_dataset,
                                                batch_size= TRAIN_BATCH_SIZE,
                                                shuffle = True,
                                                num_workers = 0)
    
    valid_dataset = BengaliDatasetTrain(folds = VALIDATION_FOLDS, 
                                        img_height= IMG_HEIGHT,
                                        img_width= IMG_WIDTH,
                                        mean = MODEL_MEAN,
                                        std = MODEL_STD)

    valid_loader = torch.utils.data.DataLoader( valid_dataset,
                                                batch_size= TEST_BATCH_SIZE,
                                                shuffle = False,
                                                num_workers = 0)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="min", 
                                                            factor = 0.3,
                                                            patience = 5,
                                                            verbose= True)
    train_model(model,train_dataset,train_loader,valid_dataset,valid_loader,optimizer,scheduler )

    

if __name__=="__main__":
    main()  