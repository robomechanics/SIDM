import torch
import numpy as np
from sidmNetworks import inputEncoder, SIDM, outputDecoder, Discriminator, sidmLosses
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
# set path to simulation file (https://github.com/robomechanics/wheeledRobotSimPybullet)
simPath = '../wheeledRobotSimPybullet'
import sys
sys.path.append(simPath)
from trajectoryDataset import trajectoryDataset
from robotStateTransformation import robotStateTransformation

class sampleLoader(object):
    def __init__ (self,data,batchSize):
        self.data = data
        self.batchSize = batchSize
        self.samples = iter([])
    def getSample(self):
        try:
            sample = next(self.samples)
        except StopIteration:
            self.samples = iter(DataLoader(self.data,shuffle=True,batch_size=self.batchSize))
            sample = next(self.samples)
        return sample

def runTraining(learningArgs,dimensionArgs,networkSizes,trainParams,data,device):
    # parse dimensions of prediction
    expertDimensionArgs = dimensionArgs[0]
    noviceDimensionArgs = dimensionArgs[1]
    latentDimensionArgs = dimensionArgs[2]
    
    # Define all networks
    expertEncoder = inputEncoder(expertDimensionArgs,latentDimensionArgs,networkSizes[0]).to(device)
    noviceEncoder = inputEncoder(noviceDimensionArgs,latentDimensionArgs,networkSizes[1]).to(device)
    sidm = SIDM(latentDimensionArgs,networkSizes[2]).to(device)
    expertDecoder = outputDecoder(expertDimensionArgs,latentDimensionArgs,networkSizes[3]).to(device)
    noviceDecoder = outputDecoder(noviceDimensionArgs,latentDimensionArgs,networkSizes[4]).to(device)
    discriminator = Discriminator(latentDimensionArgs,networkSizes[5]).to(device)
    
    # Define all optimizers
    predictionOpt = Adam(list(expertEncoder.parameters()) + list(noviceEncoder.parameters()) + list(sidm.parameters()) +
                        list(expertDecoder.parameters()) + list(noviceDecoder.parameters()),
                        lr = learningArgs[0][0], weight_decay = learningArgs[1][0])
    discriminatorOpt = Adam(discriminator.parameters(),lr = learningArgs[0][1], weight_decay = learningArgs[1][1])

    # init calculation of losses
    calcLoss = sidmLosses([expertEncoder,noviceEncoder,sidm,expertDecoder,noviceDecoder,discriminator],dimensionArgs[3],device)
    
    expertSampleLoader = sampleLoader(data[0],trainParams["expertBatchSize"])
    noviceTrainSampleLoader = sampleLoader(data[1],trainParams["noviceBatchSize"])
    #noviceTestSampleLoader = sampleLoader(data[2],trainParams["noviceBatchSize"])
    modeCounter = 0
    updateMode = 0
    for it in range(trainParams["maxIterations"]):
        print(it)
        # keep track of training modes
        while modeCounter >= trainParams['modeLengths'][updateMode]:
            modeCounter=0
            updateMode = (updateMode+1)%len(trainParams['modeLengths'])
            torch.cuda.empty_cache()
        modeCounter+=1
        # training mode 0: train prediction
        if updateMode == 0:
            # train expert with certain probability
            if np.random.rand() < trainParams["expertTrainProb"]:
                expertSample = expertSampleLoader.getSample()
                pLoss = calcLoss.getPredictionLoss(expertSample = expertSample)
            # otherwise train novice
            else:
                noviceSample = noviceTrainSampleLoader.getSample()
                pLoss = calcLoss.getPredictionLoss(noviceSample = noviceSample)
            predictionOpt.zero_grad()
            pLoss.backward()
            if np.random.rand() < trainParams["fixNoviceEncDecProb"]:
                noviceEncoder.zero_grad()
                noviceDecoder.zero_grad()
            predictionOpt.step()
        # training mode 1: train discriminator
        elif updateMode == 1:
            if np.random.rand() > 0.5:
                expertSample = expertSampleLoader.getSample()
                discriminatorLoss = calcLoss.getDiscriminatorLoss(expertSample = expertSample)
            else:
                noviceSample = noviceTrainSampleLoader.getSample()
                discriminatorLoss = calcLoss.getDiscriminatorLoss(noviceSample = noviceSample)
            discriminatorOpt.zero_grad()
            discriminatorLoss.backward()
            discriminatorOpt.step()
        # training mode 2: correspond novice
        else:
            noviceSample = noviceTrainSampleLoader.getSample()
            # flip source to make novice like expert
            correspondenceLoss = calcLoss.getCorrespondenceLoss(noviceSample)
            predictionOpt.zero_grad()
            correspondenceLoss.backward()
            predictionOpt.step()

if __name__ == '__main__':
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    experimentNumber = 1
    expertDataRootDir = 'data/system'+str(experimentNumber)+'/expert/'
    noviceDataRootDir = 'data/system'+str(experimentNumber)+'/expert/'
    csv_file_name = 'meta.csv'
    trajLength = 64
    expertData = trajectoryDataset(csv_file_name,expertDataRootDir,sampleLength=trajLength,startMidTrajectory=False,staticDataIndices=[4],device=device)
    noviceData = trajectoryDataset(csv_file_name,noviceDataRootDir,sampleLength=trajLength,startMidTrajectory=False,staticDataIndices=[4],device=device)
    expertSimParams = np.load(expertDataRootDir+'allSimParams.npy',allow_pickle=True)
    expertTerrainMapParams = expertSimParams[2]
    expertSenseParams = expertSimParams[4]
    noviceSimParams = np.load(noviceDataRootDir+'allSimParams.npy',allow_pickle=True)
    noviceTerrainMapParams = noviceSimParams[2]
    noviceSenseParams = noviceSimParams[4]
    
    """Set Dimension Arguments"""
    exampleExpertState = robotStateTransformation(expertData[0][0][0,:],terrainMap=expertData[0][4],terrainMapParams=expertTerrainMapParams,senseParams=expertSenseParams)
    expertInStateDim = exampleExpertState.getPredictionInput().shape[-1]
    expertInMapDim = exampleExpertState.getHeightMap().shape[-1]
    expertInActionDim = expertData[0][2].shape[-1]
    expertOutStateDim = expertData[0][3].shape[-1]

    exampleNoviceState = robotStateTransformation(noviceData[0][0][0,:],terrainMap=noviceData[0][4],terrainMapParams=noviceTerrainMapParams,senseParams=noviceSenseParams)
    noviceInStateDim = exampleNoviceState.getPredictionInput().shape[-1]
    noviceInMapDim = exampleNoviceState.getHeightMap().shape[-1]
    noviceInActionDim = noviceData[0][2].shape[-1]
    noviceOutStateDim = noviceData[0][3].shape[-1]

    inputLatentDim = 128
    outputLatentDim = 64
    expertDimensionArgs = [expertInStateDim,expertInMapDim,expertInActionDim,expertOutStateDim]
    noviceDimensionArgs = [noviceInStateDim,noviceInMapDim,noviceInActionDim,noviceOutStateDim]
    latentDimensionArgs = [inputLatentDim,outputLatentDim]
    mapParams = [expertTerrainMapParams,expertSenseParams,noviceTerrainMapParams,noviceSenseParams]
    dimensionArgs = [expertDimensionArgs,noviceDimensionArgs,latentDimensionArgs,mapParams]

    """Set Neural Network Sizes"""
    # sidm and discriminator
    sidmSize = [3,256]
    discriminatorSize = [1024]
    
    # expert encoder
    expertEncoderConvSize = [[8,4],[4,4]]
    expertEncoderFCSize = [128]
    expertEncoderSize = [expertEncoderConvSize,expertEncoderFCSize]

    # expert decoder size
    expertDecoderSize = [64]

    # novice encoder size
    noviceEncoderConvSize = expertEncoderConvSize
    noviceEncoderFCSize = expertEncoderFCSize
    noviceEncoderSize = [noviceEncoderConvSize,noviceEncoderFCSize]

    # novice decoder size
    noviceDecoderSize = expertDecoderSize

    networkSizes = [expertEncoderSize,noviceEncoderSize,sidmSize,expertDecoderSize,noviceDecoderSize,discriminatorSize]

    # training/ neural network parameters
    predictionLR = 0.0001
    discriminatorLR = 0.0001
    learningRate = [predictionLR,discriminatorLR]
    predictionWeightDecay = 0.00001
    discriminatorWeightDecay = 0.00001
    weight_decay=[predictionWeightDecay,discriminatorWeightDecay]
    learningArgs = [learningRate,weight_decay]

    data = [expertData,noviceData]
    correspondenceTrainParams = {"maxIterations": 500000,
                        "expertBatchSize": 2,
                        "noviceBatchSize": 2,
                        "noviceTestPercentage":0.25,
                        "expertTrainProb": 0.75, # probability of using expert data vs novice data
                        "fixNoviceEncDecProb": 0,
                        "modeLengths":[100,20,20], #num iterations prediction, discriminator, correspondence
                        "noviceSavePrefix": None,
                        "expertSavePrefix": None,
                        "testAlpha": 0.9, # used to smooth test loss
                        "trainAlpha": 0.9, # used to smooth train loss
                        "stopLength": 25000, # stop only if test loss stop decreasing for this amount of iter
                        "stopTrainLossCrit": 0.8#0.7 # stop only if train loss is half of test loss
                       }
    runTraining(learningArgs,dimensionArgs,networkSizes,correspondenceTrainParams,data,device)
