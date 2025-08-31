import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import utils as utils
import model as model

import numpy as np
from torch.utils.data import Subset

from sklearn.cluster import kmeans_plusplus

# There is known issue with having RuntimeWarnign with Macbook M4 chips. This is needed to produce repeated warnings.
# Reference: https://github.com/numpy/numpy/issues/28687
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)




def kmeans_pp (embedding, batch_size):
    embedding_np = embedding.cpu().numpy()
    _, indicies = kmeans_plusplus(embedding_np,n_clusters= batch_size)
    return indicies


def compute_gradient_embedding(DataLoader, model,  loss_fn, device):
    gradient_list = []
    model.eval()
    for X, y in DataLoader:
        X,y = X.to(device), y.to(device)

        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        loss = loss_fn(logits,preds)

        last_layer_params = model.layers[5].weight


        (last_layer_gradient,) = torch.autograd.grad(loss, last_layer_params)

        gradient_list.append(last_layer_gradient.clone().detach().flatten())
    return torch.stack(gradient_list, dim=0)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    return test_accuracy

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    QUERY_ROUNDS = 10
    BATCH_SIZE = 100

    INITIAL_LABELS = 100

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
                                batch_size= BATCH_SIZE,
                                shuffle=True
                                )
    test_dataloader =  DataLoader(dataset= test_dataset, 
                                batch_size= BATCH_SIZE, 
                                shuffle= True
                                )
    
    num_train_data = len(train_dataset)
    all_indices = list(range(num_train_data))

    np.random.shuffle(all_indices)

    labeled_indices = all_indices[:INITIAL_LABELS]
    unlabeled_indices = all_indices[INITIAL_LABELS:]

    print(f"--- Data Pool Initialization ---")
    print(f"Initial labeled pool size: {len(labeled_indices)}")
    print(f"Initial unlabeled pool size: {len(unlabeled_indices)}")
    print("---------------------------------")

    results_file = open("results_badge.txt", "a")
    results_file.write(f"\n--- New Trial Started ---\n") 

    for i in range(QUERY_ROUNDS):
        model_0 = model.ModelV0().to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
                                    model_0.parameters(),
                                    lr = 0.0001 
                                    )
        
        labeled_subset = Subset(train_dataset, labeled_indices)

        active_train_dataloader = DataLoader(dataset=labeled_subset,
                                             batch_size= BATCH_SIZE,
                                             shuffle=True
                                            )
        
        final_accuracy = train(
            epochs=100, 
            model=model_0,
            train_dataloader= active_train_dataloader,
            test_dataloader = test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        unlabled_subset = Subset(train_dataset, unlabeled_indices)

        unlabeled_dataloader = DataLoader(dataset = unlabled_subset,
                                          batch_size= 1,
                                          shuffle= False
                                          )
        computed_gradient = compute_gradient_embedding(unlabeled_dataloader, model_0, loss_fn, device)
        
        relative_indices_to_query = kmeans_pp(computed_gradient, BATCH_SIZE)

        queries_indices = [unlabeled_indices[i] for i in relative_indices_to_query]

        queried_set = set(queries_indices) # added set so it will remove duplicates faster
        
        # Below is logic to remove the new queried indices from the unlabled indices.
        new_unlabled_indices = []
        for index in unlabeled_indices:
            if index not in queried_set:
                new_unlabled_indices.append(index)
        unlabeled_indices = new_unlabled_indices

        labeled_indices.extend(queries_indices)

        # Write the result into the results_random.txt file
        log_entry = f"Round: {i+1}, Labeled Samples: {len(labeled_indices)}, Test Accuracy: {final_accuracy:.2f}%\n"
        results_file.write(log_entry)
        print(log_entry)
    results_file.close()
# Save and load:
# save_model(model_0, target_dir="models", model_name="mnist_v0.pth")
# model_loaded = utils.load_model("models/mnist_v0.pt", device=device)
# print(model_loaded.state_dict())
    
