from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from constants import *
from output_images import output_maze_images, output_non_maze_images
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2

class MazeDataset(Dataset):
    def __init__(self, num_maze_images, num_non_maze_images, train=True, transform=None): 
        output_maze_images(num_maze_images)
        output_non_maze_images(num_non_maze_images, train)
        
        self.num_maze_images = num_maze_images
        self.num_non_maze_images = num_non_maze_images
        self.transform = transform
        self.maze_images = self.load_maze_images()
        self.non_maze_images = self.load_non_maze_images()
        
    def load_maze_images(self):
        '''
        Loads maze images 
        '''
        maze_images = []
        for filename in os.listdir(MAZE_FOLDER_NAME):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image = cv2.imread(os.path.join(MAZE_FOLDER_NAME, filename))
                maze_images.append(image)
        return maze_images

    def load_non_maze_images(self):
        '''
        Loads non maze images
        '''
        non_maze_images = []
        for filename in os.listdir(NON_MAZE_FOLDER_NAME):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image = cv2.imread(os.path.join(NON_MAZE_FOLDER_NAME, filename))
                non_maze_images.append(image)
        return non_maze_images

    def __len__(self):
        return self.num_maze_images + self.num_non_maze_images

    def __getitem__(self, idx):
        if idx < self.num_maze_images:
            maze_idx = idx % len(self.maze_images)
            maze_image = self.maze_images[maze_idx]
            maze_label = torch.tensor(MAZE_LABEL)  # 0 for maze image
            if self.transform:
                maze_image = self.transform(maze_image)
            return maze_image, maze_label
        else:
            non_maze_idx = idx % len(self.non_maze_images)
            non_maze_image = self.non_maze_images[non_maze_idx]
            non_maze_label = torch.tensor(NON_MAZE_LABEL)  # 1 for non-maze image
            if self.transform:
                non_maze_image = self.transform(non_maze_image)
            return non_maze_image, non_maze_label
        
class ConvNet(nn.Module):
    '''
    Basic Conv Net
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(3380, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 3380)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)

def train(epochs, network, train_loader, optimizer):
    '''
    Trains the model for a given number of epochs and saves it
    '''
    print(f"Training maze classifier for {epochs} epochs")
    if not os.path.isdir(MODEL_FOLDER_NAME): 
        os.mkdir(MODEL_FOLDER_NAME)
    network.train()
    for _ in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                model_path = os.path.join(MODEL_FOLDER_NAME, f'{MODEL_FILE_NAME}')
                optimizer_path = os.path.join(MODEL_FOLDER_NAME, f'{OPTIMIZER_FILE_NAME}')
                torch.save(network.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optimizer_path)
    
    print("Training completed!")
    

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor()])
    train_dataset = MazeDataset(100, 100, True, transform)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    network = ConvNet()
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    train(10, network, train_dataloader, optimizer)
    