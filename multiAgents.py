# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, time, collections

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
        #time.sleep(10)
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print scores
        print legalMoves
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
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        foodCount = successorGameState.getNumFood()
        ghostScore = 12*self.getGhostScore(newPos,newGhostStates)
        minFoodDistance = self.getMinFoodDistance(newFood, newPos,currentGameState.getPacmanPosition())
        print 'minFoodDistance:'+str(minFoodDistance)
        #prevMinFoodDistance = self.getMinFoodDistance(currentGameState.getFood(), currentGameState.getPacmanPosition())
        foodScore = (30/(minFoodDistance+1))
        #foodScore = 10 * (minFoodDistance - prevMinFoodDistance)
        #finalScore =  successorGameState.getScore() - 1/foodCount+1 + 3*ghostScore + ((3)/(minFoodDistance+1))
        print "score: " + str(successorGameState.getScore())
        print "ghostScore: " + str(ghostScore)
        print "foodScore: " + str(foodScore)
        print "foodCount: " + str(foodCount)
        #if(action=='stop'):
        #    finalScore = successorGameState.getScore() - foodCount + ghostScore + foodScore
        #else:
        #finalScore =   ghostScore + foodScore - foodCount + successorGameState.getScore()
        finalScore =    ghostScore + foodScore - foodCount
        "*** YOUR CODE HERE ***"
        return finalScore

    def getFoodCount(self,newFood):
        foodCount=0
        for food in newFood:
            for pallet in food:
                if(pallet):
                    foodCount+=1
        return foodCount

    def getMinFoodDistance(self,newFood,position,currentPosition):
        minFoodDistance = 1000
        y=newFood.height
        x=newFood.width
        for x1 in xrange(0,x):
            for y1 in xrange(0,y):
                if(newFood[x1][y1]):
                    if([x1,y1]==position):
                        foodDistance=0
                    else:
                        foodDistance = manhattanDistance(position,[x1,y1])
                    if(foodDistance<minFoodDistance):
                        minFoodDistance=foodDistance
        #print newFood.width
        #for food in newFood:
        #    x=x-1
        #    y=11
        #    #print 'food length(y):'+ str(len(food))
        #    for pallet in food:
        #        y=y-1
        #        if([x,y]==currentPosition):
        #            continue
        #        if(pallet):
        #            foodDistance = manhattanDistance(position,[x,y])
        #            if(foodDistance<minFoodDistance):
        #                minFoodDistance = foodDistance
        #print 'minFoodDistance:'+str(minFoodDistance)
        return minFoodDistance

    def getGhostScore(self, position, ghostStates):
        ghostScore=0
        minDistance = 10000
        for ghost in ghostStates:
            distanceFromGhost = manhattanDistance(position,ghost.configuration.pos)
            if(distanceFromGhost<minDistance):
                minDistance = distanceFromGhost
        #for ghost in ghostStates:            
        #    if(ghost.scaredTimer>3):
        #        distanceFromGhost = manhattanDistance(position,ghost.configuration.pos)
        #        if(distanceFromGhost<minDistance):
        #            ghostScore=1
        if(ghostScore==0):
            ghostScore = (-3)/(minDistance+1)
        if(minDistance>3):
            ghostScore=0
        return ghostScore

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
        numAgents = gameState.getNumAgents()
        agentIndex=0
        maxValue = float("-inf")
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                val = self.minimizer(successor, agentIndex+1,0)
            if(val>maxValue):
                maxValue = val
                finalAction = action
        return finalAction
        util.raiseNotDefined()

    def minimizer(self, gameState, agentIndex, depth):
        minVal = float("inf")
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                if(agentIndex==(successor.getNumAgents()-1)):
                    if(depth==self.depth):
                        val = self.evaluationFunction(successor)
                        return min(val, minVal)
                    val = self.maximizer(successor, 0, depth+1)
                else:
                    val = self.minimizer(successor, agentIndex+1,depth)
            minVal = min(val,minVal)
        return minVal
    def maximizer(self, gameState, agentIndex, depth):
        maxVal = float("-inf")
        if(depth==self.depth):
            return self.evaluationFunction(gameState)
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                val = self.minimizer(successor, agentIndex+1,depth)
            maxVal = max(val, maxVal)
        return maxVal
     
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()
        agentIndex=0
        maxValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                val = self.minimizer(successor, agentIndex+1,0, alpha, beta)
            if(val>maxValue):
                maxValue = val
                finalAction = action
            if maxValue>beta:
                return maxValue
            alpha = max(maxValue, alpha)
            
        return finalAction
        util.raiseNotDefined()
    def minimizer(self, gameState, agentIndex, depth, alpha, beta):
        minVal = float("inf")
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                if(agentIndex==(successor.getNumAgents()-1)):
                    if(depth==self.depth):
                        val = self.evaluationFunction(successor)
                        minVal = min(val,minVal)
                        if minVal<alpha:
                            return minVal
                        beta = min(minVal, beta)
                        return minVal
                    val = self.maximizer(successor, 0, depth+1, alpha, beta)
                else:
                    val = self.minimizer(successor, agentIndex+1,depth, alpha, beta)
            minVal = min(val,minVal)
            if minVal<alpha:
                return minVal
            beta = min(minVal, beta)
        return minVal
    def maximizer(self, gameState, agentIndex, depth, alpha, beta):
        maxVal = float("-inf")
        if(depth==self.depth):
            return self.evaluationFunction(gameState)
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                val = self.minimizer(successor, agentIndex+1,depth, alpha, beta)
            maxVal = max(val, maxVal)
            if maxVal>beta:
                return maxVal
            alpha = max(maxVal, alpha)
        return maxVal

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
        numAgents = gameState.getNumAgents()
        agentIndex=0
        maxValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                val = self.expect(successor, agentIndex+1,0, alpha, beta)
            if(val>maxValue):
                maxValue = val
                finalAction = action
            if maxValue>beta:
                return maxValue
            alpha = max(maxValue, alpha)
            
        return finalAction
        util.raiseNotDefined()
    def expect(self, gameState, agentIndex, depth, alpha, beta):
        lstActions = gameState.getLegalActions(agentIndex)
        expVal = 0
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                if(agentIndex==(successor.getNumAgents()-1)):
                    if(depth==self.depth):
                        val = self.evaluationFunction(successor)
                        expVal = expVal + val
                        continue
                        #minVal = min(val,minVal)
                        #return minVal
                    val = self.maximizer(successor, 0, depth+1, alpha, beta)
                else:
                    val = self.expect(successor, agentIndex+1,depth, alpha, beta)
            expVal = expVal + val
        return expVal/len(lstActions)
        return 1
    def minimizer(self, gameState, agentIndex, depth, alpha, beta):
        minVal = float("inf")
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                if(agentIndex==(successor.getNumAgents()-1)):
                    if(depth==self.depth):
                        val = self.evaluationFunction(successor)
                        minVal = min(val,minVal)
                        if minVal<alpha:
                            return minVal
                        beta = min(minVal, beta)
                        return minVal
                    val = self.maximizer(successor, 0, depth+1, alpha, beta)
                else:
                    val = self.minimizer(successor, agentIndex+1,depth, alpha, beta)
            minVal = min(val,minVal)
            if minVal<alpha:
                return minVal
            beta = min(minVal, beta)
        return minVal
    def maximizer(self, gameState, agentIndex, depth, alpha, beta):
        maxVal = float("-inf")
        if(depth==self.depth):
            return self.evaluationFunction(gameState)
        lstActions = gameState.getLegalActions(agentIndex)
        for action in lstActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (successor.isLose() or successor.isWin()):
                val = self.evaluationFunction(successor)
            else:
                val = self.expect(successor, agentIndex+1,depth, alpha, beta)
            maxVal = max(val, maxVal)
            if maxVal>beta:
                return maxVal
            alpha = max(maxVal, alpha)
        return maxVal

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

