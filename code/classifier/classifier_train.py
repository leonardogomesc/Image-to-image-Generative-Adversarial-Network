# Imports
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Project Imports
from model_utils import DenseNet121, ResNet50, VGG16
from data_utils import CustomDataset, datasets, materials



# Set random seed for reproducibility
# manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)



# Select DEVICE (GPU vs CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(f"Device: {device}")


# Directories
results_dir = "results"



def main():

    # Go through all the datasets
    for dset in datasets:
        for mtrl in materials:
            try:
                print(f"Current Dataset: {dset} | Current Material: {mtrl}")

                # TODO review hyperparameters
                EPOCHS = 100

                # BATCH_SIZE = 16
                # BATCH_SIZE = 64
                BATCH_SIZE = 128

                IMG_SIZE = 64
                # IMG_SIZE = 128
                # IMG_SIZE = 224

                batch_print_interval = 1

                data_root = 'data/LivDet2015/train/'
                dataset = dset
                material = mtrl


                # Create save directories
                save_dir = os.path.join(results_dir, "classifier", dataset, material)
                if os.path.isdir(save_dir) == False:
                    os.makedirs(save_dir)



                # Build the training-pipeline
                classifier = DenseNet121(channels=3, height=64, width=64, nr_classes=1)
                
                # Put model in device
                classifier = classifier.to(device)

                # Optimizer
                optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
                

                # Load data
                train_loader = DataLoader(CustomDataset(data_root, dataset, material, IMG_SIZE, cropped=True), BATCH_SIZE, shuffle=True, num_workers=4)


                # Loss function
                loss = torch.nn.BCEWithLogitsLoss()


                # Put model in training mode
                classifier.train()


                # Variables to update
                min_loss = np.inf


                # Start loop
                for epoch in range(EPOCHS):

                    # Start timer
                    start = time.time()
                    
                    
                    # Start running loss
                    running_loss = 0.0

                    
                    # Initialise lists for labels and predictions
                    true_labels = list()
                    predictions = list()


                    for i, (images, labels) in enumerate(train_loader):
                        
                        # Pass images and labels to device
                        images = images.to(device)
                        labels = labels.to(device)

                        # Get the outputs
                        outputs = classifier(images)
                        
                        # Compute loss and perform backpropagation
                        err = loss(torch.reshape(outputs, (outputs.size(0) * outputs.size(1),)), labels)            
                        optimizer.zero_grad()
                        err.backward()
                        optimizer.step()


                        # Update running loss
                        running_loss += err.item()


                        # Update true_labels list
                        true_labels += list(labels.cpu().detach().numpy())

                        # Compute predictions
                        outputs_ = torch.sigmoid(outputs.detach())
                        
                        predictions += list(outputs_.cpu().numpy())
                        predictions = [1 if i >= 0.5 else 0 for i in predictions]


                    # Compute avg_running_loss
                    avg_running_loss = running_loss / len(train_loader.dataset)

                    # Compute accuracy
                    acc = accuracy_score(y_true=true_labels, y_pred=predictions)
                    

                    # Print metrics
                    print(f'Epoch: {epoch + 1} | Loss: {avg_running_loss} | Accuracy: {acc}')
                    print(f'Epoch Time: {time.time() - start}')


                    # Save best model
                    if avg_running_loss < min_loss:
                        torch.save(classifier.state_dict(), os.path.join(save_dir, 'best_weights_classifier.pth'))
                        print('Saved best model!')
                        min_loss = avg_running_loss

            
            except:
                print("Moving on to the next combination!")


if __name__ == '__main__':
    main()