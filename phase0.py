import pandas as pd
import numpy as np

def loadRawData(name='DJI_Train'):
    raw_data = pd.read_csv('Data/raw/'+name+'.csv')
    raw_data = raw_data.rename(index=str, columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adjclose","Volume":"volume"})
    return raw_data

def adjustData(name='DJI_Train'):
    data = loadRawData(name)
    close = data['close']
    adjustedClose = data['adjclose']
    adjustRatio = close/adjustedClose

    data['open'] = data['open']/adjustRatio
    data['close'] = data['close']/adjustRatio
    data['high'] = data['high']/adjustRatio
    data['low'] = data['low']/adjustRatio

    data.to_csv("Data/Adjusted/Adjusted_"+name+".csv",index=False)

def loadAdjustedData(name='DJI_Train'):
    adjData = pd.read_csv('Data/Adjusted/Adjusted_'+name+'.csv')
    return adjData

def adjustAndLoad(name='DJI_Train'):
    adjustData(name)
    return loadAdjustedData(name)

def loadAdjustedDevelopingData(name='DJI_Train' , percent=10):
    if percent<=0 or percent>100:
        percent = 100
    adjData = loadAdjustedData(name)
    idx = int(np.ceil((len(adjData)*percent)/100))
    subsetData = adjData.iloc[0:idx,:]
    return subsetData


if __name__ == '__main__':
    print("this is phase 0")
