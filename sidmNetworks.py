import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# set path to simulation file (https://github.com/robomechanics/wheeledRobotSimPybullet)
simPath = '../wheeledRobotSimPybullet'
import sys
sys.path.append(simPath)
from robotStateTransformation import robotStateTransformation

class inputEncoder(nn.Module):
    def __init__(self, dataDimensionArgs,latentDimensionArgs,nnSize):
        super(inputEncoder, self).__init__()
        # set size of neural network input
        self.inStateDim = dataDimensionArgs[0]
        self.inMapDim = dataDimensionArgs[1]
        self.inActionDim = dataDimensionArgs[2]
        self.inputLatentDim = latentDimensionArgs[0]

        # set sizes of neural networks
        convSize = nnSize[0]
        FCSize = nnSize[1]

        # setup input encoder & decoder conv layers
        self.convs = nn.ModuleList([])
        lastDim = [1,self.inMapDim] # num layers of channels of input, dimension of input
        for i in range(len(convSize)):
            self.convs.append(nn.Conv2d(lastDim[0],convSize[i][0],convSize[i][1]))
            lastDim = [convSize[i][0],lastDim[1]-convSize[i][1]+1]
        self.convOutputDim = lastDim[0]*lastDim[1]*lastDim[1]
        
        #set up input encoder FC layers
        self.FCs = nn.ModuleList([])
        lastDim = self.convOutputDim+self.inStateDim+self.inActionDim # size of input to first LSTM
        for i in range(len(FCSize)):
            self.FCs.append(nn.Linear(lastDim,FCSize[i]))
            lastDim = FCSize[i]
        self.FCs.append(nn.Linear(lastDim,self.inputLatentDim))

    def forward(self,data):
        # takes in robot state, map, action and predicts distribution in latent space
        rState = data[0]
        rMap = data[1]
        rAction = data[2]
        outShapePrefix = rState.shape[0:-1]
        rMap = rMap.reshape((-1,1)+(rMap.shape[-2],rMap.shape[-1]))
        rAction = rAction.reshape(-1,rAction.shape[-1])
        rState = rState.reshape(-1,rState.shape[-1])
        for i in range(len(self.convs)):
            rMap = self.convs[i](rMap)
            rMap = F.leaky_relu(rMap)
        rMap = rMap.view(-1,self.convOutputDim)
        connected = torch.cat((rMap,rState,rAction),axis=1)
        for i in range(len(self.FCs)-1):
            connected = self.FCs[i](connected)
            connected = F.leaky_relu(connected)
        connected = self.FCs[-1](connected)
        connected = F.leaky_relu(connected)
        return connected.reshape(outShapePrefix+(-1,))

class outputDecoder(nn.Module):
    def __init__(self,dataDimensionArgs,latentDimensionArgs,nnSize):
        super(outputDecoder, self).__init__()
        self.outputLatentDim = latentDimensionArgs[1]
        self.outStateDim = dataDimensionArgs[3]
        # set up fc layers
        self.FCs = nn.ModuleList([])
        lastDim = self.outputLatentDim
        for i in range(len(nnSize)):
            self.FCs.append(nn.Linear(lastDim,nnSize[i]))
            lastDim = nnSize[i]
        self.outputFC = nn.Linear(lastDim,self.outStateDim)
    def forward(self,data):
        connected = data
        for i in range(len(self.FCs)):
            connected = self.FCs[i](connected)
            connected = F.leaky_relu(connected)
        return self.outputFC(connected)

class SIDM(nn.Module):
    def __init__(self,latentDimensionArgs,nnSize):
        super(SIDM, self).__init__()
        # set size of neural network input
        self.inputLatentDim = latentDimensionArgs[0]
        self.outputLatentDim = latentDimensionArgs[1]

        #set up motion model layers
        self.latentMotionModelLSTM = nn.LSTM(input_size=self.inputLatentDim,num_layers=nnSize[0],hidden_size=nnSize[1],batch_first = True)
        self.latentMotionModelFC = nn.Linear(nnSize[1],self.outputLatentDim)
    def forward(self,latentData,prevLSTMStates=None):
        if prevLSTMStates == None:
            lstmOutput,lstmStates = self.latentMotionModelLSTM(latentData)
        else:
            lstmOutput,lstmStates = self.latentMotionModelLSTM(latentData,prevLSTMStates)
        output = F.leaky_relu(self.latentMotionModelFC(lstmOutput))
        return (output,lstmStates)

class Discriminator(nn.Module):
    def __init__(self,latentDimensionArgs,nnSize):
        super(Discriminator, self).__init__()
        # set up fully connected layers
        self.FCs = nn.ModuleList([])
        lastDim = latentDimensionArgs[0] # input latent dimension
        for i in range(len(nnSize)):
            self.FCs.append(nn.Linear(lastDim,nnSize[i]))
            lastDim = nnSize[i]
        self.outputFC = nn.Linear(lastDim,1)
    def forward(self,latentData):
        connected = latentData
        for i in range(len(self.FCs)):
            connected = self.FCs[i](connected)
            connected = F.leaky_relu(connected)
        return torch.sigmoid(self.outputFC(connected))


class sidmLosses(object):
    # initialize networks and optimizers
    def __init__(self,networks,mapParams,device):
        self.device = device
        self.expertEncoder = networks[0]
        self.noviceEncoder = networks[1]
        self.sidm = networks[2]
        self.expertDecoder = networks[3]
        self.noviceDecoder = networks[4]
        self.discriminator = networks[5]
        self.mapParams = mapParams
    def getDiscriminatorLoss(self,expertSample = None, noviceSample = None, flipSource = False):
        sourcePredictions = torch.tensor([],device=self.device)
        sourceDesired = torch.tensor([],device=self.device)
        if not expertSample is None:
            states = robotStateTransformation(expertSample[0],terrainMap=expertSample[4],terrainMapParams=self.mapParams[0],senseParams=self.mapParams[1])
            stateInput = states.getPredictionInput()
            mapInput = states.getHeightMap()
            actionInput = expertSample[2]
            predictionInputs = [stateInput,mapInput,actionInput]
            latentInputs = self.expertEncoder(predictionInputs)
            expertSourcePrediction = self.discriminator(latentInputs)
            sourcePredictions = torch.cat((sourcePredictions,expertSourcePrediction),dim=0)
            sourceDesired = torch.cat((sourceDesired,torch.ones_like(expertSourcePrediction)),dim=0)
        if not noviceSample is None:
            states = robotStateTransformation(noviceSample[0],terrainMap=noviceSample[4],terrainMapParams=self.mapParams[2],senseParams=self.mapParams[3])
            stateInput = states.getPredictionInput()
            mapInput = states.getHeightMap()
            actionInput = noviceSample[2]
            predictionInputs = [stateInput,mapInput,actionInput]
            latentInputs = self.noviceEncoder(predictionInputs)
            noviceSourcePrediction = self.discriminator(latentInputs)
            sourcePredictions = torch.cat((sourcePredictions,noviceSourcePrediction),dim=0)
            sourceDesired = torch.cat((sourceDesired,torch.zeros_like(noviceSourcePrediction)),dim=0)
        if flipSource:
            sourceDesired = 1-sourceDesired
        discriminatorLoss = F.binary_cross_entropy(sourcePredictions,sourceDesired)
        return discriminatorLoss
    def getCorrespondenceLoss(self,noviceSample):
        return self.getDiscriminatorLoss(noviceSample=noviceSample,flipSource=True)
    def getPredictionLoss(self,expertSample = None,noviceSample = None):
        allPredictions = torch.tensor([],device=self.device)
        allTruths = torch.tensor([],device=self.device)
        if not expertSample is None:
            prediction,truth = self.singleStepPrediction(expertSample,forExpert=True)
            allPredictions = torch.cat((allPredictions,prediction),dim=0)
            allTruths = torch.cat((allTruths,truth),dim=0)
        if not noviceSample is None:
            prediction,truth = self.singleStepPrediction(noviceSample,forExpert=False)
            allPredictions = torch.cat((allPredictions,prediction),dim=0)
            allTruths = torch.cat((allTruths,truth),dim=0)
        return F.mse_loss(allPredictions,allTruths)
    def singleStepPrediction(self,sample,forExpert):
        if forExpert:
            terrainMapParams = self.mapParams[0]
            senseParams = self.mapParams[1]
            encoderDecoder = [self.expertEncoder,self.expertDecoder]
        else:
            terrainMapParams = self.mapParams[2]
            senseParams = self.mapParams[3]
            encoderDecoder = [self.noviceEncoder,self.noviceDecoder]
        states = robotStateTransformation(sample[0],terrainMap=sample[4],terrainMapParams=terrainMapParams,senseParams=senseParams)
        stateInput = states.getPredictionInput()
        mapInput = states.getHeightMap()
        actionInput = sample[2]
        predictionInputs = [stateInput,mapInput,actionInput]
        latentInputs = encoderDecoder[0](predictionInputs)
        latentOutputs,_ = self.sidm(latentInputs)
        predictions = encoderDecoder[1](latentOutputs)
        truth = states.getRelativeState(sample[3])
        return predictions, truth