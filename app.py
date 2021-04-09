import json
import os
import numpy as np
import pandas as pd
from flask import Flask, request, Response
from flask_cors import CORS
from Utils import printColumns, loadFile, findKey, readData, timeProcess, matchData, writeFile
from collections import Counter
import scipy.stats as scs
import datetime

app = Flask(__name__)
# 解决跨域问题
CORS(app, supports_credentials=True)

# 存储进度条进度数据
progress_data = {'getCorr': 0, 'train': 0, 'predict': 0}

# 声明全局变量，用户选择的数据
VIBParamsTrain = {}  # 用于训练的振动参数信息，{文件名：飞参}
VIBParamsVerify = {}  # 用于验证的振动参数信息
FPParamsTrain = {}  # 用于训练的飞参信息
FPParamsVerify = {}  # 用于验证的飞参信息
SubSampling = False  # 是否欠采样
# 设置文件访问基本路径
baseURL = "data/"


###########################################数据录入########################################
@app.route('', methods=['POST'])
def searchRepeat():
    """
    筛选出重复的变量名进行记录
    :return:
    """
    global GreenItem
    params = []
    input_data = json.loads(str(request.get_data().decode()))
    files = input_data.get("files")
    for file in files:
        param = printColumns(baseURL + file)
        param.remove("TIME")
        params = params + param
    count = dict(Counter(params))
    GreenItem = [key for key, value in count.items() if value > 1]


@app.route('/paramList', methods=['GET'])
def paramList():
    """
    获取所选文件里面的的所有参数，以字典的形式返回
    请求参数为fileName--文件路径
    :return:
    """
    result = {}
    fileName = request.args.get("fileName")
    filePath = baseURL + fileName
    params = printColumns(filePath)
    params.remove("TIME")
    for param in params:
        if param in GreenItem:
            result[param] = 1
        else:
            result[param] = 0
    return Response(json.dumps(params), mimetype='application/json')


@app.route('/dataLoad', methods=['POST'])
def dataLoad():
    """
    将用户选择的训练参数和测试参数保存到全局变量中，方便后面查找参数的路径
    :return:
    """
    global SubSampling
    global VIBParamsTrain
    global FPParamsTrain
    global FPParamsVerify
    global VIBParamsVerify
    input_data = json.loads(str(request.get_data().decode()))
    FPParamsTrain = input_data.get("FPParams_train")
    VIBParamsTrain = input_data.get("VIBParams_train")
    FPParamsVerify = input_data.get("FPParams_verify")
    VIBParamsVerify = input_data.get("VIBParams_verify")
    SubSampling = input_data.get("SubSampling")
    if SubSampling is None:
        SubSampling = False
    else:
        SubSampling = eval(SubSampling)
    return Response(json.dumps('录入数据成功'), mimetype='application/json')


########################################数据预处理#################################
@app.route('/vibParamList', methods=['GET', 'POST'])
def vibParamList():
    """
    展示用户在前面选择的训练集振动参数，用户可以在其中选择振动参数进行相关性训练，可以重复选择
    :return:
    """
    vibs = []
    for vibParam in VIBParamsTrain.keys():
        if type(VIBParamsTrain[vibParam]) is list:
            vibs = vibs + VIBParamsTrain[vibParam]
        else:
            vibs.append(VIBParamsTrain[vibParam])
    return Response(json.dumps(list(set(vibs))), mimetype='application/json')


@app.route('/getCorr', methods='GET')
def getCorr():
    """
    根据用户选择的振动参数做相关性分析
    :return:
    """
    vibName = request.args.get("vibName")
    progress_data['getCorr'] = 0  # 将进度条置零
    # 判断所选择的振动参数是否做过相关性分析
    if vibName + "_FP_col" in os.listdir("Savedfile") and vibName + "_FP_corr" in os.listdir("Savedfile"):
        vib_FP_highCorr_FP = loadFile('Savedfile/' + vibName + '_FP_corr')
        vib_FP_highCol_FP = loadFile('Savedfile/' + vibName + '_FP_col')
        progress_data['getCorr'] = float(100)  # 进度条直接到100%
    else:
        repeat = []
        corr_list = []  # 初始化相关性list
        col_list = []  # 初始化变量名list
        zeroCor_191207 = loadFile("Savedfile/zeroCor_191207")
        vibFolder = baseURL + findKey(VIBParamsTrain, vibName)[0]  # 查找需要做相关性分析的参数所在文件路径
        vibData, vibTime = readData(vibFolder, vibName)  # 读取振动数据及相应时间
        tempFPParams = FPParamsTrain
        # 先遍历一次，统计飞参个数
        FpNumber = 0
        for file in FPParamsTrain.keys():
            FpNumber += len(FPParamsTrain[file])
        i = 1  # 初始化计数器
        for file in tempFPParams.keys():
            df = pd.read_table(baseURL + file, sep='\t')
            time_i = timeProcess(df['TIME'].tolist())
            for FPParam in tempFPParams[file]:
                if (FPParam in zeroCor_191207):  # 如果某个变量在zeroCor_191207中
                    print(FPParam + " Break!")  # 告知并跳过该变量的计算
                elif ("Accel" in FPParam):  # 如果某个变量包括"Accel"在变量名称中，该变量为加速度传感器数据
                    print(FPParam + " Break!")  # 告知并跳过该变量的计算
                elif (FPParam in repeat):  # 如果有重复变量且已经做过相关性分析
                    print(FPParam + " Break!")  # 告知并跳过该变量的计算
                else:
                    files = findKey(FPParamsTrain, FPParam)
                    # 如果有重复的飞参，则进行拼接
                    if len(files) > 1:
                        repeat.append(FPParam)  # 记录处理过的重复飞参

                    jCol = df[FPParam].fillna(0, inplace=True).tolist()
                    jCol_matched = matchData(vibTime, time_i, jCol)  # 将飞参数据与振动数据时间轴匹配
                    corrJ = scs.pearsonr(jCol_matched, vibData)[0]  # 计算该飞参数据与振动响应数据的相关性
                    corr_list.append(corrJ)  # 将计算的相关性加入corr_list中
                    col_list.append(FPParam)  # 将变量名称加入col_list
                    print("Correlation between " + FPParam + " and " + str(vibName) + " is " + str(
                        corrJ))  # 告知变量名与相关性
                num_progress = round(i * 100 / FpNumber, 2)
                i += 1
                progress_data['getCorr'] = num_progress  # 更新进度条
        print(" Fetching complete!")  # 告知飞参变量的读取完毕
        # 按绝对值从大到小进行排序
        corr = np.array(corr_list)
        col = np.array(col_list)
        vib_FP_highCorr_FP = corr[np.argsort(np.abs(corr))[::-1]].tolist()
        vib_FP_highCol_FP = col[np.argsort(np.abs(corr))[::-1]].tolist()
        # 按绝对值从大到小进行保存
        writeFile("Savedfile/" + vibName + "_FP_corr", vib_FP_highCorr_FP)
        writeFile("Savedfile/" + vibName + "_FP_col", vib_FP_highCol_FP)
    result = {"corr_list": vib_FP_highCorr_FP, "col_list": vib_FP_highCol_FP}
    return Response(json.dumps(result), mimetype='application/json')


####################################################模型训练########################################
@app.route('/vibNameList', methods=['GET'])
def vibNameList():
    """
    振动参数选择，只能选择做过相关性分析并保存在Savedfile下的飞参
    :return:
    """
    vibNames = []
    for i in os.listdir("Savedfile"):
        if "FP_col" in i:
            vibNames.append(i.split("_")[0])
    return Response(json.dumps(vibNames), mimetype='application/json')


@app.route('/getNum', methods=['GET'])
def getNum():
    """
    根据用户输入的num和选择的振动参数，返回相关性最高的num个飞参
    :return:
    """
    global vibNameforTrain
    vibNameforTrain = request.args.get('vibName')
    num = request.args.get('num')
    vib_high_FP_corr = loadFile('Savedfile/' + vibNameforTrain + '_FP_corr')
    vib_high_FP_col = loadFile('Savedfile/' + vibNameforTrain + '_FP_col')
    # 选择相关性最高的num个飞参
    num = int(num)
    FPPre = vib_high_FP_col[0:num]
    result = {"col": vib_high_FP_col, "checked": FPPre}
    return result


@app.route('/train', methods=['POST'])
def train():
    """
    训练模型
    :return:
    """
    global LRreg
    global SVRreg
    global ANNreg
    global LR_meta
    global vib_FP_highCol_FP
    global start
    global spendTime
    progress_data['train'] = 0
    input_data = json.loads(str(request.get_data().decode()))
    modelPath = input_data.get("modelPath")
    vib_FP_highCol_FP = input_data.get("FPSelected")  # 用户选择的用于训练的飞参
    start = datetime.datetime.now()
    length = 500
    # 需要预测的测点的频率
    select_fre = range(0, 200)
    # 振动参数加载

    vib_value_train, fp_value_train = readVIB_FPData(vibNameforTrain, VIBParamsTrain, vib_FP_highCol_FP, FPParamsTrain,
                                                     baseURL, length, 'train', SubSampling)
    vib_value_verify, fp_value_verify = readVIB_FPData(vibNameforTrain, VIBParamsVerify, vib_FP_highCol_FP,
                                                       FPParamsVerify, baseURL, length, 'train', SubSampling)

    vib_abs_train, vib_ang_train = FFTvibData(vib_value_train, length)
    vib_abs_verify, vib_ang_verify = FFTvibData(vib_value_verify, length)
    vib_ary_train = vib_abs_train[:, select_fre]
    vib_ary_verify = vib_abs_verify[:, select_fre]
    # 对转换后的频域信号进行滑动时间窗移动平均
    vib_ary_train = timeWindows(vib_ary_train, length=10)
    progress_data['train'] = progress_data['train'] + 5.00  # 65%
    vib_ary_verify = timeWindows(vib_ary_verify, length=10)
    progress_data['train'] = progress_data['train'] + 5.00  # 70%
    if modelPath is None or modelPath == "":
        '''
        线性回归
        '''
        LRreg = LinearRegression().fit(fp_value_train, vib_ary_train)
        vib_pred_LR_verify = LRreg.predict(fp_value_verify)
        '''
        支持向量机回归
        '''
        SVRreg = []
        base_SVRreg = make_pipeline(SVR(kernel='rbf', C=100, tol=0.1, gamma=0.1,
                                        epsilon=0.5, max_iter=1000))
        base_process = progress_data['train']
        for i in range(len(select_fre)):
            SVRreg.append(sklearn.base.clone(base_SVRreg).fit(fp_value_train, vib_ary_train[:, i]))
            num_progress = round((i + 1) * 100 * 0.25 / len(select_fre), 2) + base_process  # 95%
            progress_data['train'] = num_progress
        vib_pred_SVR_verify = np.array([SVRreg[i].predict(fp_value_verify) for i in range(len(select_fre))]).T

        '''
        人工神经网络
        '''
        ANNreg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,),
                                                              alpha=0.001, learning_rate_init=0.001,
                                                              random_state=1, max_iter=100))
        ANNreg.fit(fp_value_train, vib_ary_train)
        vib_pred_ANN_verify = ANNreg.predict(fp_value_verify)
        '''
        算法集合体
        预测及结果展示
        '''
        LR_meta = []
        base_LR_meta = LinearRegression()
        for i in range(len(select_fre)):
            vib_pred_ALL_verify = np.hstack((vib_pred_LR_verify[:, i].reshape(-1, 1),
                                             vib_pred_SVR_verify[:, i].reshape(-1, 1),
                                             vib_pred_ANN_verify[:, i].reshape(-1, 1)))
            LR_meta.append(sklearn.base.clone(base_LR_meta).fit(vib_pred_ALL_verify, vib_ary_verify[:, i]))
    else:
        with open('model/' + modelPath, 'rb') as f:
            LRreg = pickle.load(f)  # 从f文件中提取出模型赋给clf2
            SVRreg = pickle.load(f)
            ANNreg = pickle.load(f)
            LR_meta = pickle.load(f)
        '''
       线性回归
       '''
        LRreg = LRreg.fit(fp_value_train, vib_ary_train)
        vib_pred_LR_verify = LRreg.predict(fp_value_verify)
        '''
        支持向量机回归
        '''
        SVRreg = []
        base_SVRreg = make_pipeline(SVR(kernel='rbf', C=100, tol=0.1, gamma=0.1,
                                        epsilon=0.5, max_iter=1000))
        base_process = progress_data['train']
        for i in range(len(select_fre)):
            SVRreg.append(sklearn.base.clone(base_SVRreg).fit(fp_value_train, vib_ary_train[:, i]))
            num_progress = round((i + 1) * 100 * 0.25 / len(select_fre), 2) + base_process  # 95%
            progress_data['train'] = num_progress
        vib_pred_SVR_verify = np.array([SVRreg[i].predict(fp_value_verify) for i in range(len(select_fre))]).T
        progress_data['train'] = num_progress
        '''
        人工神经网络
        '''
        ANNreg.fit(fp_value_train, vib_ary_train)
        vib_pred_ANN_verify = ANNreg.predict(fp_value_verify)
        '''
        算法集合体
        预测及结果展示
        '''
        LR_meta = []
        base_LR_meta = LinearRegression()
        for i in range(len(select_fre)):
            vib_pred_ALL_verify = np.hstack((vib_pred_LR_verify[:, i].reshape(-1, 1),
                                             vib_pred_SVR_verify[:, i].reshape(-1, 1),
                                             vib_pred_ANN_verify[:, i].reshape(-1, 1)))
            LR_meta.append(sklearn.base.clone(base_LR_meta).fit(vib_pred_ALL_verify, vib_ary_verify[:, i]))
    end = datetime.datetime.now()
    spendTime = str((end - start).seconds)
    progress_data['train'] = 100.00
    result = {"VIBParamsTrain": VIBParamsTrain, "FPParamsTrain": FPParamsTrain, "vibNameforTrain": vibNameforTrain,
              "fpNameforTrain": vib_FP_highCol_FP, "beginTime": str(start.strftime('%Y-%m-%d %H:%M:%S')),
              "spendTime": spendTime}
    return Response(json.dumps(result), mimetype='application/json')


def readVIB_FPData(vib_Name, VIBParams, FPnames, FPParams, baseURL, length, methodName, SubSampling=False):
    vibFile = findKey(VIBParams, vib_Name)[0]
    vibFolder = baseURL + vibFile
    vib_value, vib_time = readData(vibFolder, vib_Name)
    vib_value = np.array(vib_value[:(len(vib_value) - len(vib_value) % length)])
    vib_value = np.reshape(vib_value, (-1, length))
    vib_time = np.array(vib_time[:(len(vib_time) - len(vib_time) % length)])
    vib_time = np.reshape(vib_time, (-1, length))
    vib_time = vib_time[:, 0]
    FPnames = list(reversed(FPnames))                          #待确定



if __name__ == '__main__':
    app.run()
