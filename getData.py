import numpy as np
import os
# set path to simulation file (https://github.com/robomechanics/wheeledRobotSimPybullet)
simPath = '../wheeledRobotSimPybullet'
import sys
sys.path.append(simPath)
from parallelSimDataset import gatherData


if __name__ == '__main__':
    runSystem1 = True
    runSystem2 = True
    # parameters for parallel processing
    numParallelSims = 12
    trajectoryLength = 64
    numTrajectoriesPerSim = np.ceil(500000/numParallelSims/trajectoryLength)
    # parameters consistent between systems
    simParams = {"timeStep":1./500.,
                "stepsPerControlLoop":50,
                "numSolverIterations":300,
                "gravity":-10,
                "contactBreakingThreshold":0.0001,
                "contactSlop":0.0001,
                "moveThreshold":0.1,
                "maxStopMoveLength":25}
    terrainMapParams = {"mapWidth":300, # width of matrix
                    "mapHeight":300, # height of matrix
                    "widthScale":0.1, # each pixel corresponds to this distance
                    "heightScale":0.1}
    senseParams = {"senseDim":[5,5], # width (meter or angle) and height (meter or angle) of terrain map or point cloud
                    "senseResolution":[300,300], # array giving resolution of map output (num pixels wide x num pixels high)
                    "senseType":-1, # 0 for terrainMap, 1 for lidar depth image, 2 for lidar point cloud
                    "sensorPose":[[0,0,0],[0,0,0,1]]} # pose of sensor relative to body
    dataRootDir = 'data/'
    if not os.path.isdir(dataRootDir):
        os.mkdir(dataRootDir)
    if runSystem1:
        # parameters for expert and novice
        expertRobotParams = {"maxThrottle":60,
                            "maxSteerAngle":0.8,
                            "susOffset":-0.002,
                            "susLowerLimit":-0.02,
                            "susUpperLimit":0.00,
                            "susDamping":50,
                            "susSpring":10000,
                            "traction":1,
                            "massScale":1.5}
        noviceRobotParams = {"maxThrottle":30,
                            "maxSteerAngle":0.5,
                            "susOffset":-0.001,
                            "susLowerLimit":-0.02,
                            "susUpperLimit":0.00,
                            "susDamping":100,
                            "susSpring":50000,
                            "traction":1.25,
                            "masScale":1.0}
        terrainParams = {"AverageAreaPerCell":1,
                        "cellPerlinScale":5,
                        "cellHeightScale":0.35,
                        "smoothing":0.7,
                        "perlinScale":2.5,
                        "perlinHeightScale":0.1}
        experimentRootDir = dataRootDir + 'system1/'
        if not os.path.isdir(experimentRootDir):
            os.mkdir(experimentRootDir)
        noviceRootDir = experimentRootDir+'novice/'
        if not os.path.isdir(noviceRootDir):
            os.mkdir(noviceRootDir)
        np.save(noviceRootDir+'allSimParams.npy',[simParams,noviceRobotParams,terrainMapParams,terrainParams,senseParams])
        print('System 1 novice parameters set at: ' + noviceRootDir)
        expertRootDir = experimentRootDir+'expert/'
        if not os.path.isdir(expertRootDir):
            os.mkdir(expertRootDir)
        np.save(expertRootDir+'allSimParams.npy',[simParams,expertRobotParams,terrainMapParams,terrainParams,senseParams])
        # gather data for novice
        gatherData(numParallelSims,numTrajectoriesPerSim,trajectoryLength,noviceRootDir,True)
        # gather data for expert
        gatherData(numParallelSims,numTrajectoriesPerSim,trajectoryLength,expertRootDir,True)
    if runSystem2:
        # parameters for expert and novice
        noviceRobotParams = {"maxThrottle":30,
                            "maxSteerAngle":0.5,
                            "susOffset":-0.0,
                            "susLowerLimit":-0.01,
                            "susUpperLimit":0.003,
                            "susDamping":100,
                            "susSpring":50000,
                            "traction":1.0,
                            "massScale":1}
        expertRobotParams = {"maxThrottle":40,
                            "maxSteerAngle":0.75,
                            "susOffset":-0.002,
                            "susLowerLimit":-0.02,
                            "susUpperLimit":0.00,
                            "susDamping":1,
                            "susSpring":75,
                            "traction":1.25,
                            "massScale":1.5}
        terrainParams = {"AverageAreaPerCell":1,
                        "cellPerlinScale":5,
                        "cellHeightScale":0.9,
                        "smoothing":0.7,
                        "perlinScale":2.5,
                        "perlinHeightScale":0.1}
        experimentRootDir = dataRootDir + 'system2/'
        if not os.path.isdir(experimentRootDir):
            os.mkdir(experimentRootDir)
        noviceRootDir = experimentRootDir+'novice/'
        if not os.path.isdir(noviceRootDir):
            os.mkdir(noviceRootDir)
        np.save(noviceRootDir+'allSimParams.npy',[simParams,noviceRobotParams,terrainMapParams,terrainParams,senseParams])
        print('System 1 novice parameters set at: ' + noviceRootDir)
        expertRootDir = experimentRootDir+'expert/'
        if not os.path.isdir(expertRootDir):
            os.mkdir(expertRootDir)
        np.save(expertRootDir+'allSimParams.npy',[simParams,expertRobotParams,terrainMapParams,terrainParams,senseParams])
        print('System 2 expert parameters set at: ' + expertRootDir)
        # gather data for novice
        gatherData(numParallelSims,numTrajectoriesPerSim,trajectoryLength,noviceRootDir,True)
        # gather data for expert
        gatherData(numParallelSims,numTrajectoriesPerSim,trajectoryLength,expertRootDir,True)
