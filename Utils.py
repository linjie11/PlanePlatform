import pickle
import pandas as pd


def printColumns(filename_):
    """
    返回输入文件名（filename_)中的所有变量名
    返回值类型为list
    """
    f_ = open(filename_, "r")  # 打开目标文件
    colname = f_.readline().split("\t")  # 读取变量名称
    colname[-1] = colname[-1].strip()  # 去掉空格
    # print (colname)
    f_.close()
    return (colname)  # 输出变量名称


def loadFile(loadName):
    """
    读取地址（loadName）的文件
    返回值类型为string
    """
    f_ = open(loadName, "rb")  # 以已读方式打开loadName地址的文件
    returnFile = pickle.load(f_)  # 读取打开的文件，并将其存入returnFile变量
    f_.close()  # 关闭打开的文件

    return (returnFile)  # 输出returnFile


def findKey(map, value):
    """
    获取参数在哪几个文件当中
    """
    fileNames = []
    for fileName in map.keys():
        if value in map[fileName]:
            fileNames.append(fileName)
    return fileNames


def readData(filename_, colname_, subsampling=False):
    """
    使用pandas读取对应参数数据和时间
    """
    df = pd.read_table(filename_, sep='\t')
    data = df[colname_].fillna(0, inplace=True)
    data = data.tolist()
    time = df['TIME'].tolist()
    ms = timeProcess(time)
    return data, ms


def timeProcess(time):
    """
    将时间转换成毫秒
    :param time: 原始时间
    :return: 转换后的毫秒
    """
    ms = []
    for item in time:
        timeList = item.split(":")
        timei = 0  # 初始化以毫秒为单位的时间计数器
        for i in range(4):  # 读取由时、分、秒、毫秒构成的timeList
            if (i < 3):
                timei += int(timeList[i]) * 60 ** (2 - i)  # 转换时分秒
            else:
                timei *= 1000
                timei += int(timeList[i])  # 转换为毫秒
        ms.append(timei)  # 将timei加入输出list
    return ms


def matchData(timeA, timeB, aviB):
    """
    根据基准时间匹配数据，保证aviB长度和timeA一致
    :param timeA: 基准时间
    :param timeB: 需要匹配的时间
    :param aviB: 根据时间进行匹配的数据
    :return: 匹配时间后的数据
    """
    counter = 0  # 计数器初始化为0，计数器代表aviB所在位置
    returnList = []  # 初始化输出list为空
    for i in range(len(timeA)):  # 以timeA的序列为基准进行匹配
        if counter + 1 >= len(timeB):  # 如果计数大小已经超过B的长度
            counter -= 1  # 计数器停止增长
        elif timeA[i] == timeB[counter + 1]:  # 或者如果timeB的下一位置可匹配timeA的当前位置
            counter += 1  # 计数器加1
        elif timeA[i] == timeB[counter + 1] + 1:  # 或者如果timeB的下一位置加1可匹配timeA的当前位置（time的取值频率不固定，如果对照实际数据理解会更加清晰）
            counter += 1  # 计数器加1
        returnList.append(aviB[counter])  # 输出list加入当前位置B的变量
    return returnList  # 返回输出list

def writeFile(writeName, writeFile):
    '''
    将文件（writeFile）存入文件地址（writeName）
    无返回
    '''
    f_ = open(writeName, "wb")  # 在writeName地址创建新文件，如果新文件已经存在将打开并进行修改
    pickle.dump(writeFile, f_)  # 将变量writeFile写入打开的文件
    f_.close()  # 关闭文件
