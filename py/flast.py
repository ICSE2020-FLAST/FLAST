from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import os
import random
import time
import warnings


###############################################################################


def getDataPoints(path, exclude=".java"):
    dataPointsList = []
    for dataPointName in os.listdir(path):
        if dataPointName[-len(exclude):] != exclude:
            continue
        with open(os.path.join(path, dataPointName)) as fileIn:
            dp = fileIn.read()
            dataPointsList.append(dp)
    return dataPointsList


def getFlakyInfoList(projectName, flakyInfoPath, flakyPath, exclude=".java"):
    projectName = projectName.split("/")[1]
    flakyInfoDict = defaultdict(list)
    with open(flakyInfoPath) as fileIn:
        h = next(fileIn)
        for line in fileIn:
            proj, className, method, flaky = line.split(",")
            if proj == projectName:
                className = className.strip().split("/")[-1]
                key = "{}-{}{}".format(className, method.strip(), exclude)
                flakyInfoDict[key].append(flaky.strip())

    flakyInfoList = []
    for dataPointName in os.listdir(flakyPath):
        if dataPointName[-len(exclude):] != exclude:
            continue
        flakyInfo = flakyInfoDict.get(dataPointName, "-")
        flakyInfoList.append(set(flakyInfo))
    return flakyInfoList


def getDataPointsInfo(projectBasePath, projectName):
    # get list of tokenized test methods
    projectPath = os.path.join(projectBasePath, projectName)
    flakyPath = os.path.join(projectPath, "flakyMethods")
    nonFlakyPath = os.path.join(projectPath, "nonFlakyMethods")
    dataPointsFlaky = getDataPoints(flakyPath)
    dataPointsNonFlaky = getDataPoints(nonFlakyPath)
    projectClass = os.path.join(projectBasePath, projectName.split("/")[0])
    flakyInfoPath = os.path.join(projectClass, "flakiness.csv")
    flakyInfoList = getFlakyInfoList(projectName, flakyInfoPath, flakyPath)

    return dataPointsFlaky, dataPointsNonFlaky, flakyInfoList


def vectorizeDataPointsSRP(dataPoints, reduceDim=True, dim=0, eps=0.33):
    countVec = CountVectorizer()
    Z_full = countVec.fit_transform(dataPoints)
    if reduceDim:
        if dim <= 0:
            dim = johnson_lindenstrauss_min_dim(Z_full.shape[0], eps=eps)
        srp = SparseRandomProjection(n_components=dim)
        Z = srp.fit_transform(Z_full)
        return Z
    else:
        return Z_full


###############################################################################
# CLASSIFICATION EXP

def classifyFlakyTestExperiment(kNN, trainData, testData, trainLabels, testLabels, threshold):
    # training
    t0 = time.perf_counter()
    kNN.fit(trainData, trainLabels)
    t1 = time.perf_counter()

    # testing
    p0 = time.perf_counter()
    predictLabels = kNN.predict(testData)
    p1 = time.perf_counter()

    if threshold > 0:
        # attempt to increase precision
        for i, pred in enumerate(predictLabels):
            if pred == 1:  # if predicted flaky
                dp = np.array(testData[i]).reshape(1, -1)
                dist, ind = kNN.kneighbors(dp)  # get neighbors
                f, nf = 0, 0
                for l in range(len(ind[0])):
                    j, d = ind[0][l], dist[0][l]
                    if trainLabels[j] == 1:
                        f += d
                    else:
                        nf += d
                try:
                    if (f / (f + nf)) < threshold:
                        predictLabels[i] = 0
                except:
                    predictLabels[i] = 0

    # compute metrics
    res = {}
    # efficiency
    res["totTrainTime"] = t1 - t0
    res["totPredTime"] = p1 - p0
    res["avgPredTime"] = res["totPredTime"] / len(testData)

    # effectiveness
    warnings.filterwarnings("error")  # to catch warnings, e.g., "prec set to 0.0"
    try:
        res["f-measure"] = f1_score(testLabels, predictLabels)
    except:
        res["f-measure"] = "-"
    try:
        res["precision"] = precision_score(testLabels, predictLabels)
    except:
        res["precision"] = "-"
    try:
        res["recall"] = recall_score(testLabels, predictLabels)
    except:
        res["recall"] = "-"
    warnings.resetwarnings()  # warnings are no more errors

    res["accuracy"] = accuracy_score(testLabels, predictLabels)
    tn, fp, fn, tp = confusion_matrix(testLabels, predictLabels).ravel()
    res["tp"] = tp
    res["fp"] = fp
    res["fn"] = fn
    res["tn"] = tn

    return res, predictLabels


def classifyFlakyCauseExperiment(kNN, testData, trainLabels, predictLabels, trainInfo, testInfo):
    predictInfo = []
    indices = kNN.kneighbors(testData, return_distance=False)
    for i, nIndices in enumerate(indices):
        if predictLabels[i] == 1:
            kLabelsCount = defaultdict(int)
            for nInd in nIndices:
                if trainLabels[nInd] == 1:
                    for x in trainInfo[nInd]:
                        kLabelsCount[x] += 1
            candidates = list(kLabelsCount.keys())
            random.shuffle(candidates)
            try:
                predict = max(candidates, key=(lambda k: kLabelsCount[k]))
            except ValueError:
                predict = "?"
            predictInfo.append((i, predict))

    infoPred = defaultdict(int)
    infoTot = defaultdict(int)
    infos = {i for s in trainInfo for i in s}
    if "-" in infos:
        infos.remove("-")

    accuracyScore, n = 0, 0
    for i, p in predictInfo:
        if predictLabels[i] == 1 and p != "-":
            n += 1
            infoTot[p] += 1
            if p in testInfo[i]:
                accuracyScore += 1
                infoPred[p] += 1
    if n > 0:
        accuracyScore /= n
    else:
        accuracyScore = "-"

    info = {}
    for i in infos:
        if infoTot[i] > 0:
            info[i] = infoPred[i] / infoTot[i]

    return accuracyScore, info


###############################################################################


def printDict(s, d):
    print(s)
    for k, v in d.items():
        print(" {}:".format(k), v)
    print()
