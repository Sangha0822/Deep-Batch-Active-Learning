import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import utils as utils
import model as model



def train(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer, device):

    for epoch in range(epochs):
        # Training loop:
        model.train() 
        

        total_train_loss = 0
        correct_train_predictions = 0
        total_train_samples = 0

        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)


            train_logits = model(X)
            

            loss = loss_fn(train_logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_train_loss += loss.item() * X.size(0) 
            

            train_preds = torch.argmax(train_logits, dim=1)
            correct_train_predictions += (train_preds == y).sum().item()
            total_train_samples += X.size(0)


        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = (correct_train_predictions / total_train_samples) * 100 

        # Testing loop:
        model.eval() 
        

        total_test_loss = 0
        correct_test_predictions = 0
        total_test_samples = 0

        with torch.inference_mode(): 
            for batch_idx, (Z, a) in enumerate(test_dataloader):
                Z, a = Z.to(device), a.to(device)

                test_logits = model(Z)
                

                test_loss_batch = loss_fn(test_logits, a)
                

                total_test_loss += test_loss_batch.item() * Z.size(0)

                test_preds = torch.argmax(test_logits, dim=1)
                correct_test_predictions += (test_preds == a).sum().item()
                total_test_samples += Z.size(0)
            

            avg_test_loss = total_test_loss / total_test_samples
            test_accuracy = (correct_test_predictions / total_test_samples) * 100

        print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.5f} | Train Acc: {train_accuracy:.2f}% | Test Loss: {avg_test_loss:.5f} | Test Acc: {test_accuracy:.2f}%")
        
        
        if train_accuracy >= 99.0:
            print(f"Training accuracy reached 99.0% at epoch {epoch}. Stopping training for this model.")
            break 

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )

    test_dataset = datasets.MNIST(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    )

    train_dataloader = DataLoader(dataset= train_dataset, 
                                batch_size= 64, 
                                shuffle=True
                                )
    test_dataloader =  DataLoader(dataset= test_dataset, 
                                batch_size= 64, 
                                shuffle= True
                                )


    model_0 = model.ModelV0().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
                                model_0.parameters(),
                                lr = 0.001 
                                )
    
    train(
        epochs=100, 
        model=model_0,
        train_dataloader=train_dataloader,
        test_dataloader = test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

# Save and load:
# save_model(model_0, target_dir="models", model_name="mnist_v0.pth")
# model_loaded = utils.load_model("models/mnist_v0.pt", device=device)
# print(model_loaded.state_dict())
    
