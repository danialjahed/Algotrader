import pandas as pd
import numpy as np
import talib
import phase0

# phase0.adjustData(name='DJI_Train')
# phase0.adjustData(name='DJI_Test')
trainData = phase0.loadAdjustedDevelopingData(name='DJI_Train', percent=15)
testData = phase0.loadAdjustedDevelopingData(name='DJI_Test', percent=10)


if __name__ == '__main__':

    dataRSI = []
    for i in range(2,22):
        dataRSI.append(talib.RSI(trainData['adjclose'].as_matrix(),i))

    dataSMA = []
    dataSMA.append(talib.SMA(trainData['adjclose'].as_matrix(),50))
    dataSMA.append(talib.SMA(trainData['adjclose'].as_matrix(),200))
    SMALable = []
    for i in range(len(dataSMA[0])):
        if np.isnan(dataSMA[0][i]) or np.isnan(dataSMA[1][i]):
            SMALable.append(-1)
        elif dataSMA[0][i]>dataSMA[1][i]:
            SMALable.append(1)
        else:
            SMALable.append(1)
    dataSMA.append(SMALable)

    dataRSI = pd.DataFrame(dataRSI).transpose()
    dataRSI.columns = ['interval1','interval2','interval3','interval4','interval5',
                        'interval6','interval7','interval8','interval9','interval10',
                        'interval11','interval12','interval13','interval14','interval15',
                        'interval16','interval17','interval18','interval19','interval20',]
    dataSMA = pd.DataFrame(dataSMA).transpose()
    dataSMA.columns = ['SMA50','SMA200','trend']

    outPutData_DataFrame = pd.concat([trainData['close'],dataRSI, dataSMA], axis=1)

    outPutData_DataFrame.to_csv("Data/geneticInput/DJI_Train.csv")

    inputData = []
    for i in range(len(dataRSI)):
        for j in range(20):
            inputData.append({"RSIValue":dataRSI.iloc[i,j], "RSIInterval":j+1, "trend":dataSMA.iloc[i,2]})

    inputData_DataFrame = pd.DataFrame(inputData)
    # print(inputData_DataFrame.head())
