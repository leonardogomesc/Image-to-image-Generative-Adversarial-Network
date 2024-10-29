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

                # Batch-size
                BATCH_SIZE = 128

                # Image size (H X W)
                IMG_SIZE = 64


                # Data Directories
                data_root = 'data/LivDet2015/test/'
                dataset = dset
                material = mtrl


                # Classifier Directories
                # classifier_dir = os.path.join(results_dir, "classifier", dataset, material)
                # classifier_dir = os.path.join(results_dir, "classifier-daug", dataset, material)
                classifier_dir = os.path.join(results_dir, "classifier-cdaug", dataset, material)


                # Build the training-pipeline
                classifier = DenseNet121(channels=3, height=64, width=64, nr_classes=1)
                
                # Load weights
                classifier.load_state_dict(torch.load(os.path.join(classifier_dir, "best_weights_classifier.pth"), map_location=device))

                # Move classifier into device
                classifier = classifier.to(device)
                
                # Put model in evaluation mode
                classifier.eval()
                

                # Load data
                test_loader = DataLoader(CustomDataset(data_root, dataset, material, IMG_SIZE, resize_ratio=1.1, data_aug=False, aug_cond=False, cropped=False, test=True), BATCH_SIZE, shuffle=True, num_workers=4)
                

                # Start loop
                with torch.no_grad():

                    
                    # Initialise lists for labels and predictions
                    true_labels = list()
                    predictions = list()


                    for _, (images, labels) in enumerate(test_loader):
                        
                        # Pass images and labels to device
                        images = images.to(device)
                        labels = labels.to(device)

                        # Get the outputs
                        outputs = classifier(images)

                        # Update true_labels list
                        true_labels += list(labels.cpu().detach().numpy())

                        # Compute predictions
                        outputs_ = torch.sigmoid(outputs.detach())
                        
                        predictions += list(outputs_.cpu().numpy())
                        predictions = [1 if i >= 0.5 else 0 for i in predictions]


                    # Compute accuracy
                    acc = accuracy_score(y_true=true_labels, y_pred=predictions)
                    

                    # Print metrics
                    print(f'Accuracy (Test Set): {acc}')


            
            except:
                print("Moving on to the next combination!")


if __name__ == '__main__':
    main()