# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)  
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        # Consider the distance to closest food, distancet to ghost, and the food left for evaluation score 
        foodList = newFood.asList()
        foodDistance = min([manhattanDistance(newPos, food) for food in foodList]) if foodList else 0
        ghostDistance = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        
        ghostScore = -5000 if ghostDistance <= 1 else 0
        foodScore = -5 * foodDistance if foodDistance else 0

        return foodScore + ghostScore + -200 * len(foodList)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        minmaxValue = []
        legalMoves = gameState.getLegalActions(agentIndex=0)
        for move in legalMoves:
            value = self.minmax_search(gameState.generateSuccessor(0, move), agent=0, depth=0)
            minmaxValue.append((value, move))
 
        return max(minmaxValue)[1]

    
    def minmax_search(self, gameState, agent, depth):
        def terminal_state():
            """ If there's no legal move or finish searching for depth return score"""
            legalMoves = gameState.getLegalActions(agent)
            
            if not legalMoves or depth == self.depth:
                return self.evaluationFunction(gameState)

            return None

        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        terminal = terminal_state()
        if terminal is not None:
            return terminal

        legalMoves = gameState.getLegalActions(agent)
        
        if agent == 0:
            pacmanValue = []
            for move in legalMoves:
                pacmanValue.append((self.minmax_search(gameState.generateSuccessor(agent, move), agent + 1, depth), move))
            
            return max(pacmanValue)[0]

        else:
            ghostValue = []
            for move in legalMoves:
                ghostValue.append((self.minmax_search(gameState.generateSuccessor(agent, move), agent + 1, depth), move))
            
            return min(ghostValue)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """  
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        value = (float("-inf"), "None")
        alpha, beta = float("-inf"), float("inf")
        legalMoves = gameState.getLegalActions(agentIndex=0)
        for move in legalMoves:
            value = max(value, (self.alphabeta(gameState.generateSuccessor(0, move), 1, 0, alpha, beta), move))
            if value[0] > beta:
                return value[1]
            alpha = max(value[0], alpha)

        return value[1]


    def alphabeta(self, gameState, agent, depth, alpha, beta):
        def terminal_state():
            """ If there's no legal move or finish searching for depth return score"""        
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            return None

        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        terminal = terminal_state()
        if terminal is not None:
            return terminal

        legalMoves = gameState.getLegalActions(agent)
        if agent == 0:
            value = float("-inf")
            for move in legalMoves:
                value = max(value, self.alphabeta(gameState.generateSuccessor(agent, move), agent + 1, depth, alpha, beta))
                if value > beta:
                    return value
                alpha = max(value, alpha)
            return value
        else:
            value = float("inf")
            for move in legalMoves:
                value = min(value, self.alphabeta(gameState.generateSuccessor(agent, move), agent + 1, depth, alpha, beta))
                if value < alpha:
                    return value
                beta = min(value, beta)
            return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        value = []
        legalMoves = gameState.getLegalActions(agentIndex=0)
        for move in legalMoves:
            expectValue = self.expectMinMax(gameState.generateSuccessor(0, move), 1, 0)
            value.append((expectValue, move))
        
        maxValue = max(value)
        maxIndex = filter(lambda i: value[i] == maxValue, range(len(value)))
        return value[random.choice(maxIndex)][1]

    def expectMinMax(self, gameState, agent, depth):
        def terminal_state():
            """ If there's no legal move or finish searching for depth return score"""        
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            return None

        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        terminal = terminal_state()
        if terminal is not None:
            return terminal

        legalMoves = gameState.getLegalActions(agent)
        if agent == 0:
            value = float("-inf")
            for move in legalMoves:
                value = max(value, 
                            self.expectMinMax(gameState.generateSuccessor(agent, move), agent + 1, depth))
            return float(value)
        else:
            value = [self.expectMinMax(gameState.generateSuccessor(agent, move),
                     agent + 1, depth) for move in legalMoves]

            return float(sum(value)) / float(len(value))
        

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: 
        - feature list: currentScore, distance to closest food, distance to closest non scared ghost,
                        distance to closest capsule, number of food left, number of capsule left
        - I use two different approach(linear combinations or non-linear combinations) to choose
          which one is better.
        -> I found out that both non-linear combination and linear exist makes the score higher
    """

    score = currentGameState.getScore()
    newCapsules = currentGameState.getCapsules()
    pacman = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    
    score -= (20 * pow(len(newCapsules), 2) + 2 * len(foodList))
    score -= 0.5 * sum((manhattanDistance(food, pacman) for food in foodList))
    score -= 2 * sum((manhattanDistance(capsule, pacman) for capsule in newCapsules))

    for ghost in newGhostStates:
        distance = float(manhattanDistance(ghost.getPosition(), pacman))
        if ghost.scaredTimer > 0:
            if distance <= ghost.scaredTimer:
                score += 200 * pow((1.0 / distance), 2)
        else:
            if distance == 0:
                score -= 5000
            elif 1 <= distance <= 10:
                score -= 10 * (1.0 / distance)
    return score


# Abbreviation
better = betterEvaluationFunction
