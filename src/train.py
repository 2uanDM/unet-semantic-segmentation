import os
import time
import torch.nn as nn
import torch
from model.unet_enhanced import ResNet50_UNET
from model.loss import CEDiceLoss
from dataset.dataset import NeoPolypDataset
from torch.utils.data import DataLoader, random_split

def weights_init(model):
    if isinstance(model, nn.Linear):
        # Xavier Distribution
        torch.nn.init.xavier_uniform_(model.weight)

def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

# Train function for each epoch
def train(
        model,
        loss_function,
        device, 
        optimizer, 
        train_dataloader, 
        valid_dataloader,
        learing_rate_scheduler, 
        epoch, 
        display_step
    ):
    print(f"Start epoch #{epoch+1}, learning rate for this epoch: {learing_rate_scheduler.get_last_lr()}")
    start_time = time.time()
    train_loss_epoch = 0
    test_loss_epoch = 0
    last_loss = 999999999
    model.train()
    for i, (data,targets) in enumerate(train_dataloader):
        
        # Load data into GPU
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        # Backpropagation, compute gradients
        loss = loss_function(outputs, targets.long())
        loss.backward()

        # Apply gradients
        optimizer.step()
        
        # Save loss
        train_loss_epoch += loss.item()
        if (i+1) % display_step == 0:
#             accuracy = float(test(test_loader))
            print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {:.4f}'.format(
                epoch + 1, (i+1) * len(data), len(train_dataloader.dataset), 100 * (i+1) * len(data) / len(train_dataloader.dataset), 
                loss.item()))
                  
    print(f"Done epoch #{epoch+1}, time for this epoch: {time.time()-start_time}s")
    train_loss_epoch/= (i + 1)
    
    # Evaluate the validation set
    model.eval()
    with torch.no_grad():
        for data, target in valid_dataloader:
            data, target = data.to(device), target.to(device)
            test_output = model(data)
            test_loss = loss_function(test_output, target)
            test_loss_epoch += test_loss.item()
            
    test_loss_epoch/= (i+1)
    
    return train_loss_epoch , test_loss_epoch

if __name__ == "__main__":
    # Model
    model = ResNet50_UNET(n_classes=3)
    model.apply(weights_init)
    model.cuda()
    
    # Loss function
    loss_function = CEDiceLoss(weights=torch.tensor([0.1, 0.4, 0.5]).cuda())
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler
    learing_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print(os.getcwd())
    
    # Dataset
    train_dataset = NeoPolypDataset(img_dir=os.path.join(os.getcwd(), 'train', 'train'),
                                    gt_img_dir=os.path.join(os.getcwd(), 'train_gt', 'train_gt'),
                                    status='train')
    
    train_dataset, valid_dataset = random_split(train_dataset, [int(len(train_dataset)*0.8), len(train_dataset)-int(len(train_dataset)*0.8)])
    
    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=25, shuffle=False, num_workers=2)
    
    # Train
    epochs = 200
    display_step = 10
    for epoch in range(epochs):
        train_loss, valid_loss = train(model, loss_function, torch.device('cuda'), optimizer, train_dataloader, valid_dataloader, learing_rate_scheduler, epoch, display_step)
        print(f"Epoch #{epoch+1} train loss: {train_loss}, valid loss: {valid_loss}")
        learing_rate_scheduler.step()
        save_model(model, optimizer, f"model-{epoch+1}.pth")
    save_model(model, optimizer, f"model-final.pth")