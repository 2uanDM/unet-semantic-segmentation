import os 
import sys
import time 
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import wandb
import argparse

from src.model.resnet50unet import ResNet50_UNET
from src.dataset.dataset import NeoPolypDataset
from src.model.loss import CEDiceLoss, calc_dice_score
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, Compose, InterpolationMode
from torch import optim
from torch.optim import lr_scheduler as lrs

def __save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def __load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def train(model, optimizer, loss_fn, train_dataloader ,val_dataloader, lr_scheduler, epoch, device):
    print(f'Start epoch #{epoch + 1}, lr = {lr_scheduler.get_last_lr()}')
    
    start_time = time.time()
    train_loss_epoch = 0
    train_acc_epoch = 0
    train_dice_score_list = []
    val_dice_score_list = []
    
    # Training
    model.train()
    for i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad() # Reset gradient
        train_output = model(X) # Forward
        loss = loss_fn(train_output, y) # Calculate loss
        loss.backward() # Backward
        optimizer.step() # Update weight
        
        # Compute dice score
        val_dice_score = calc_dice_score(train_output, y)
        train_dice_score_list.append(val_dice_score.item())
        
        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch + 1}], Step [{i + 1}/{len(train_dataloader.dataset)}] \tLoss: {loss.item():.4f}, Dice score: {val_dice_score.item():.4f}')
        
        train_loss_epoch /= (i + 1)
        train_dice_score_epoch = sum(train_dice_score_list) / len(train_dice_score_list)
    
    
    # Validation
    model.eval()
    with torch.no_grad():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            test_output = model(X)
            test_loss = loss_fn(test_output, y)
            test_loss_epoch += test_loss.item()
            
            # Compute the dice score
            val_dice_score = calc_dice_score(test_output, y)
            val_dice_score_list.append(val_dice_score.item())
            
        test_loss_epoch /= (i+1)
        test_dice_score_epoch = sum(val_dice_score_list) / len(val_dice_score_list)
    
    print(f"Done epoch #{epoch + 1}, time for this epoch: {time.time() - start_time}s")
    
    return train_loss_epoch, test_loss_epoch, train_dice_score_epoch, test_dice_score_epoch
        
    
def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', '-e', type=int, default=100)
    
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    
    parser.add_argument('--split_ratio', '-s', type=float, default=0.8)
    
    parser.add_argument('--wandb', '-w', default=True)
    
    parser.add_argument('--wandb_key', '-wk', type=str)
    
    args = parser.parse_args()
    
    # Wandb
    if args.wandb:
        wandb.login(key=args.wandb_key)
        name = f'dl-{args.epochs}-{args.batch_size}-{args.lr}'
        wandb.init(project='DL-Assignment', name=name)
    
    # Model 
    model = ResNet50_UNET(n_classes=3)
    model = nn.DataParallel(model) 
    model.to(device)
    
    # Dataset 
    transform = Compose([
        Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        PILToTensor()
        
    ])
    train_dataset = NeoPolypDataset(
        os.path.join(args.data_path, 'train', 'train'), 
        os.path.join(args.data_path, 'train_gt', 'train_gt'), 
        transform=transform)
    
    train_size = int(args.split_ratio * len(train_dataset))
    test_size = len(train_dataset) - train_size
    
    train_set, valid_set = random_split(train_dataset, [train_size , test_size])
    
    # Dataloader
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    
    # Lr scheduler
    lr_scheduler = lrs.StepLR(optimizer, step_size=4, gamma=0.6)
    
    # Loss function
    weights = torch.Tensor([[0.4, 0.55, 0.05]]).cuda()
    loss_fn = CEDiceLoss(weights)
    

    # Train
    train_loss_array = []
    test_loss_array = []
    last_loss = 9999999999999
    for epoch in range(args.epochs):
        train_loss_epoch = 0
        test_loss_epoch = 0
        train_loss_epoch, test_loss_epoch, train_dice, test_dice = train(
            model=model, 
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            device=device)
        
        if test_loss_epoch < last_loss:
            __save_model(model, optimizer, 'model.pth')
            last_loss = test_loss_epoch
            
        lr_scheduler.step()
        train_loss_array.append(train_loss_epoch)
        test_loss_array.append(test_loss_epoch)
        wandb.log({"Train loss": train_loss_epoch, "Valid loss": test_loss_epoch, 'Train dice':train_dice, 'Test dice': test_dice})
    

if __name__ == '__main__':
    main()