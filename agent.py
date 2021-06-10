import multiprocessing as mp
import random
from collections import deque

import torch

from games import AIOne, AITwo, AIThree, AIFour, AIFive, AISix, AISeven, AIEight, AINine, Game
from utils.plotter import plot
from models import LinearModel, ModelTrainer, TunnedLinearModel

import pygame

MAX_MEMORY = 200_000
BATCH_SIZE = 2000
LR = 0.001


class Agent:
    def __init__(self, inputSize: int, hiddenLayerSize: int = 256):
        # We store the number of games played, this is used later to determine
        # whether we take a random action or use the model to predict the next
        # action.
        self.nGames = 0

        # Used in conjunction with the nGames to take a random action or a calculated
        # one.
        self.episilon = 0

        # The discount rate used to calculate the new Q in the model. This value
        # is a multiplier to the reward for each step. The default values for ML problems
        # are in between [0.9, 0.95].
        self.gamma = 0.9

        # Acts as the memory for our learning. It stores, for each state, the action that was taken,
        # the reward it got for that action, the next state and wheter the game was done or not.
        self.memory = deque(maxlen=MAX_MEMORY)

        # Create the Q Net model, using the inputSize as the first layer, 256 as the hidden layer and 3 as the output layer.
        # The number 3 is fixed, since the game only needs 3 values to take action.

        if inputSize < 100:
            self.model = LinearModel(inputSize, hiddenLayerSize, 3)
        else:
            self.model = TunnedLinearModel(inputSize, hiddenLayerSize, 3)

        # Create the training model. using the defined learning rate and gamma values.
        self.trainer = ModelTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        # Append the states to the memory. Will automatically delete the first
        # states if the size of the deque exceeds MAX_MEMORY.
        self.memory.append((state, action, reward, next_state, done))

    def trainLongMemory(self):
        # Check if we have enough samples in our memory to reach the defined batch size. If not, use the complete
        # memory as the input for the train step.
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE)
        else:
            miniSample = self.memory

        # Retrieve from the sample each input as a vector as pass them to the training step.
        states, actions, rewards, next_states, dones = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)

    def trainShortMemory(self, state, action, reward, next_state, done):
        # Instead of using the memory to pass multiple samples to training step, the short memory train
        # uses only the current state and results as the input.
        self.trainer.trainStep(state, action, reward, next_state, done)

    def getAction(self, state, training=True):
        # Calculating the tradeoff between exploration and exploitation.
        # This will determine the likelihood of the snake taking a random action every step.
        # The chance of taking a random action decreases with each game played.
        self.epsilon = 80 - self.nGames

        finalMove = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon and training:
            # If the random number is below our randomness threshold, randomly choose a direction
            # for the snake to go to.
            move = random.randint(0, 2)
            finalMove[move] = 1
        else:
            # Transform the network input to a tensor, of type float.
            state0 = torch.tensor(state, dtype=torch.float)

            # Predict the next move using the current state.
            prediction = self.model(state0)

            # Since the ouputs are given as float values, we need to take the highest one and use that
            # the predicted move the snake should make.
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove


def train(modelName: str, model: Game, size: int, iters: int = 400, visual=False, hiddenLayerSize=256):
    # Variables we need to store in order to be able to plot the results.
    initIters = iters

    # The achieved score for each game played.
    plotScores = []
    # The mean score, calculated after each game.
    plotMeanScores = []
    # The total score, the sum of all scores.
    totalScore = 0
    # The highest score ever reached.
    recordScore = 0

    # Create the game.
    game = model(size, 1, visual)

    # Create the agent.
    agent = Agent(game.getInputSize(), hiddenLayerSize)

    while iters > 0:
        # Get current game state.
        currentState = game.getState()

        # Calculate the next action based on the current state.
        finalMove = agent.getAction(currentState)

        # Take the action and retrieve the results of that action.
        reward, done, score = game.move(finalMove)

        # Get the new state of the game, after we've taken that action.
        newState = game.getState()

        agent.trainShortMemory(currentState, finalMove, reward, newState, done)
        agent.remember(currentState, finalMove, reward, newState, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.nGames += 1
            agent.trainLongMemory()

            if score > recordScore:
                recordScore = score
                agent.model.save(modelName + ".pth")

            plotScores.append(score)
            totalScore += score
            plotMeanScores.append(totalScore / agent.nGames)
            print(
                f"Model: {modelName}\tGame {agent.nGames}\tScore {score}\tRecord {recordScore}")
            if visual:
                plot(plotScores, plotMeanScores)

            iters -= 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                iters = 0

    with open(f"./results//training/{modelName}.txt", "w") as f:
        f.write(str(size))
        f.write("\n")
        f.write(str(initIters))
        f.write("\n")
        f.write(str(recordScore))
        f.write("\n")
        f.write(" ".join([str(i) for i in plotScores]))
        f.write("\n")
        f.write(" ".join([str(i) for i in plotMeanScores]))
        f.write("\n")


def play(model: str, game: Game, size: int, hiddenLayerSize=256, iters=-1):

    initIters = iters
    game = game(size, 1, 1)
    inputSize = game.getInputSize()

    agent = Agent(inputSize, hiddenLayerSize)
    agent.model.load(model)

    nGames = 0
    recordScore = 0
    totalScore = 0

    plotScores = []
    plotMeanScores = []

    playing = True
    while playing:
        # Get current game state.
        currentState = game.getState()

        # Calculate the next action based on the current state.
        finalMove = agent.getAction(currentState, False)

        # Take the action and retrieve the results of that action.
        reward, done, score = game.move(finalMove)

        # Get the new state of the game, after we've taken that action.
        newState = game.getState()

        if done:
            # train long memory, plot result
            game.reset()
            agent.nGames += 1

            if score > recordScore:
                recordScore = score

            plotScores.append(score)
            totalScore += score
            plotMeanScores.append(totalScore / agent.nGames)
            print(
                f"Game {agent.nGames}\tScore {score}\tRecord {recordScore}")
            plot(plotScores, plotMeanScores)

            if iters != -1:
                iters -= 1
                if iters < 0:
                    playing = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False

    with open(f"./results/playing/{model.split('.')[0]}.txt", "w") as f:
        f.write(str(size))
        f.write("\n")
        f.write(str(initIters-iters))
        f.write("\n")
        f.write(str(recordScore))
        f.write("\n")
        f.write(" ".join([str(i) for i in plotScores]))
        f.write("\n")
        f.write(" ".join([str(i) for i in plotMeanScores]))
        f.write("\n")


if __name__ == "__main__":
    models = [
        # ["aione_32_32", AIOne, 32],
        # ["aione_longbatch_32_32", AIOne, 32],
        # ["aitwo_17_17", AITwo, 32],
        # ["aitwo_32_32", AITwo, 32],
        # ["aitwo_longbatch_32_32", AITwo, 32],
        # ["aithree_32_32", AIThree, 32],
        # ["aifour_32_32", AIFour, 32],
        # ["aifive_32_32", AIFive, 32],
        # ["aisix_32_32", AISix, 32],
        # ["aiseven_v2_tunned_32_32", AISeven, 32],
        # ["aiseven_v2_17_17", AISeven, 32],
        ["aiseven_v2_32_32", AISeven, 32],
        # ["aiseven_32_32", AISeven, 32],
        # ["aieight_tunned_32_32", AIEight, 32],
        # ["aieight_longbatch_tunned_32_32", AIEight, 32],
        # ["aieight_17_17", AIEight, 32],
        # ["aieight_32_32", AIEight, 32],
        # ["ainine_17_17", AINine, 32],
        # ["ainine_32_32", AINine, 32],
    ]
    for i in range(len(models)):
        modelName, model, size = models[i]
        modelName = f"{modelName}.pth"
        play(modelName, model, size, iters=200)
    #     p = mp.Process(target=train, args=(modelName, model, size, 800, True))
    #     p.start()
    # train('aieight_32_32', AIEight, 32, 800, True)
    # train('ainine_32_32', AINine, 32, 800, True)
    # train('aitwo_17_17', AITwo, 17, 800, True)
    # play('aiseven_v2_17_17.pth', AISeven, 32)
