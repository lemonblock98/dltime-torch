import numpy as np

def calSingleCorrelation(a, b):
    """计算a b两个信号的余弦相关性
    """
    numerator = 0
    dedominator = 0
    tempA = 0
    tempB = 0
    for i, j in zip(a, b):
        numerator += i*j
        tempA += i*i
        tempB += j*j
    dedominator = np.sqrt(tempA*tempB)
    return np.abs(numerator)/dedominator

def calCorrelation(a, b, windowSize, stepSize):
    """按滑动窗口，计算a b信号的余弦相似性
    """
    corr = []
    for i in range(0, len(a), stepSize):
        corr.append(calSingleCorrelation(a[i: i+windowSize], b[i: i+windowSize]))
    return corr

def activitySplit(dataXYZ, windowSize, stepSize, corrThresholdLength):
    """以余弦相关性来切割事件
    """
    joinWindowSize = windowSize  # 两个窗口拼接成一个时间的阈值
    # folderPath = r"D:\研一\土壤扰动\syf\dig"
    # csvPath = folderPath + "\\" + csvName
    # dataXYZ = pd.read_csv(csvPath, header= 0)
    corrXYs = calCorrelation(dataXYZ.iloc[:, 0], dataXYZ.iloc[:, 1], windowSize, stepSize) 
    activityPosList = []
    isFirst = True
    
    corrThresholdList = corrXYs[: corrThresholdLength]
    corrThresholdListCopy = corrThresholdList.copy()
    corrThresholdListCopy.sort()
    percent25 = corrThresholdListCopy[int(0.25*corrThresholdLength)]
    percent75 = corrThresholdListCopy[int(0.75*corrThresholdLength)]
    
    coef = 3
    corrThreshold = percent25 - coef*(np.abs(percent75 - percent25))
    #corrThreshold = np.abs(percent75 + coef*(np.abs(percent75 - percent25)))
    
    for i, corrXY in enumerate(corrXYs):
        if np.abs(corrXY) <= corrThreshold:
            if isFirst:
                start = i * stepSize
                end = start + windowSize
                isFirst = False
            else:
                if i*stepSize - end <= joinWindowSize:
                    end = i*stepSize + windowSize
                else:
                    activityPosList.append([start, end])
                    start = i * stepSize
                    end = start + windowSize
        corrThresholdList.pop(0)
        corrThresholdList.append(corrXY)
        corrThresholdListCopy = corrThresholdList.copy()
        corrThresholdListCopy.sort()
        percent25 = corrThresholdListCopy[int(0.25*corrThresholdLength)]
        percent75 = corrThresholdListCopy[int(0.75*corrThresholdLength)]
        corrThreshold = percent25 - coef*(np.abs(percent75 - percent25))
        # corrThreshold = np.abs(percent75 + coef*(np.abs(percent75 - percent25)))

    # print(csvName, len(activityPosList))
    return activityPosList


def cal_sig_diff(sig, diffStep=1):
    """对某一信号按step计算绝对值差分"""
    return [np.abs(sig[i+diffStep]-sig[i]) for i in range(0, len(sig)-diffStep, diffStep)]


def cal_window_mean(sig, windowSize=16, stepSize=8, diffStep=1):
    """给定信号, 按windowSize的窗口大小, stepSize为步长, 按diffStep做差分, 计算差分均值"""
    ans = []
    for i in range(0, len(sig), stepSize):
        ans.append(np.mean(cal_sig_diff(sig[i: i+windowSize], diffStep)))
    
    return ans


def activitySplit_diffMean(dataZ, windowSize=16, stepSize=8, meanThresholdLength=200):
    """使用绝对差分的均值来切割事件"""
    joinWindowSize = windowSize
    meanZs = cal_window_mean(dataZ, windowSize, stepSize, step=2)
    
    activityPosList = []
    isFirst = True

    meanThresholdList = meanZs[: meanThresholdLength]
    meanThresholdListCopy = meanThresholdList.copy()
    meanThresholdListCopy.sort()
    percent25 = meanThresholdListCopy[int(0.25*meanThresholdLength)]
    percent75 = meanThresholdListCopy[int(0.75*meanThresholdLength)]

    coef = 3
    meanThreshold = percent75 + coef*(percent75 - percent25)
    for i, meanZ in enumerate(meanZs):
        if meanZ >= meanThreshold:
            if isFirst:
                start = i * stepSize
                end = start + windowSize
                isFirst = False
            else:
                if i*stepSize - end <= joinWindowSize:
                    end = i*stepSize + windowSize
                else:
                    activityPosList.append([start, end])
                    start = i * stepSize
                    end = start + windowSize
        
        meanThresholdList.pop(0)
        meanThresholdList.append(meanZ)
        meanThresholdListCopy = meanThresholdList.copy()
        meanThresholdListCopy.sort()
        percent25 = meanThresholdListCopy[int(0.25*meanThresholdLength)]
        percent75 = meanThresholdListCopy[int(0.75*meanThresholdLength)]
        meanThreshold = percent75 + coef*(percent75 - percent25)
        # corrThreshold = np.abs(percent75 + coef*(np.abs(percent75 - percent25)))

    # print(csvName, len(activityPosList))
    return activityPosList
