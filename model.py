import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearModel(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        # Creates the network layers with the given sizes.
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        # Function required to train the model and take next step.
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, fileName="model.pth"):
        # Saves the current model.
        modelFolderPath = "./model"
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        filePath = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), filePath)


class ModelTrainer:
    def __init__(self, model, lr, gamma):
        # Initializes all the attributes with the given values.
        self.model = model
        self.lr = lr
        self.gamma = gamma

        # Create optimizer using the model parameters and learning rate.
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # The criteria to calculate the model's accuracy is MSE.
        self.criterion = nn.MSELoss()

    def trainStep(self, states, actions, rewards, next_states, done):
        # Transform the given inputs into tensors.
        state = torch.tensor(states, dtype=torch.float)
        next_state = torch.tensor(next_states, dtype=torch.float)
        action = torch.tensor(actions, dtype=torch.long)
        reward = torch.tensor(rewards, dtype=torch.float)

        # Check if the input are in the form of (n, x), meaning index followed by
        # value. If not, change the dimesion to (1, x) using the unsqueeze function
        # from PyTorch.
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Get the predicted Q values for the current state.
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            qNew = reward[idx]
            if not done[idx]:
                qNew = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = qNew

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
