from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import time

import flast


if __name__ == "__main__":
    print("RQ - CATEGORY IDENTIFICATION EFFECTIVENESS")

    # DATASETS
    projectBasePath = "datasets"
    projectList = [
        "PalombaEtAl/ant-ivy",  # 764 vs 411
        "PalombaEtAl/apache-derby",  # 84 vs 10094
        "PalombaEtAl/apache-hbase",  # 277 vs 2647
        "PalombaEtAl/apache-hibernate-orm",  # 127 vs 3703
        "PalombaEtAl/apache-hive",  # 106 vs 541
        "PalombaEtAl/apache-karaf",  # 120 vs 265
        "PalombaEtAl/apache-nutch",  # 184 vs 73
        "PalombaEtAl/apache-pig",  # 1268 vs 3528
        "PalombaEtAl/apache-qpid",  # 314 vs 2950
        "PalombaEtAl/apache-wicket",  # 216 vs 1866
        "PalombaEtAl/lucene-solr",  # 7 vs 5327
        #
        # "DeFlaker/handlebars",  # 7 vs 523
        # "DeFlaker/logback",  # 11 vs 1041
        # "DeFlaker/okhttp",  # 32 vs 1199
        # "DeFlaker/tachyon",  # 7 vs 609
        #
        "iDFlakies/activiti",  # 20 vs 33
        "iDFlakies/apache-hadoop",  # 68 vs 1053
        "iDFlakies/apache-incubator-dubbo",  # 21 vs 486
        "iDFlakies/elastic-job-lite",  # 10 vs 775
        "iDFlakies/http-request",  # 28 vs 140
        "iDFlakies/java-websocket",  # 52 vs 436
        "iDFlakies/retrofit",  # 9 vs 415
        "iDFlakies/vertx-rabbitmq-client",  # 7 vs 37
        "iDFlakies/wildfly",  # 43 vs 72
    ]

    # FLAST NEAREST NEIGHBORS PARAMETERS
    kNNParams = {
        "algorithm": "brute",
        "metric": "cosine",
        "n_neighbors": 7,
        "weights": "distance",
        "n_jobs": 1
    }
    dim = 0  # -1: no reduction; 0: JL with error eps
    eps = 0.33  # JL error in (0, 1)
    threshold = 0.5  # similarity threshold in [0, 1]; values used for results are in {0.5, 0.95}

    # EXPERIMENT PARAMETERS
    numKFold = 10  # number of folds for KFold Cross Validation

    # DO NOT TOUCH THIS
    # parameters transformation from FLAST parameters to parameters for experiments
    if threshold < 0.5 or threshold > 1:
        print("FLAST threshold must be in [0.5, 1]")
        exit()
    if threshold == 0.5:
        threshold = 0
    thresholdName = "t-0.5" if threshold == 0 else "t-{}".format(threshold)
    reduceDim = False if dim < 0 else True

    resultsPath = "results/rq-afc_category_precision/{}".format(thresholdName)
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/rq-afc_category_precision/"):
        os.makedirs("results/rq-afc_category_precision")
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    for projectName in projectList:
        p0 = time.time()

        # FLAST setup
        v0 = time.perf_counter()
        kNN = KNeighborsClassifier(
            algorithm=kNNParams["algorithm"],
            metric=kNNParams["metric"],
            n_neighbors=kNNParams["n_neighbors"],
            weights=kNNParams["weights"],
            n_jobs=kNNParams["n_jobs"])

        dataPointsFlaky, dataPointsNonFlaky, flakyInfoList = flast.getDataPointsInfo(projectBasePath, projectName)

        print("#"*80)
        flast.printDict("CLASSIFIER PARAMETERS", kNNParams)
        print("*"*80)
        print(" Project Name:", projectName)
        print(" Number of Flaky Tests:", len(dataPointsFlaky))
        print(" Total Number of Tests:", len(dataPointsFlaky) + len(dataPointsNonFlaky))

        # create dataPointsList
        dataPoints = dataPointsFlaky + dataPointsNonFlaky
        Z = flast.vectorizeDataPointsSRP(dataPoints, reduceDim=reduceDim, dim=dim, eps=eps)

        dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
        dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
        dataInfoList = np.array(flakyInfoList + ["-"]*len(dataPointsNonFlaky))
        v1 = time.perf_counter()

        resListDict = defaultdict(list)
        resListInfo = defaultdict(list)

        successFold = 0
        kf = StratifiedKFold(n_splits=numKFold, shuffle=True, random_state=42)
        for kFold, (trnIdx, tstIdx) in enumerate(kf.split(dataPointsList, dataLabelsList)):
            trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
            trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
            trainInfo, testInfo = dataInfoList[trnIdx], dataInfoList[tstIdx]
            if sum(trainLabels) == 0 or sum(testLabels) == 0:
                print("Skipping fold...")
                print(" Flaky Train Tests", sum(trainLabels))
                print(" Flaky Test Tests", sum(testLabels))
                continue
            elif len(trainData) < kNNParams["n_neighbors"]:
                print("Skipping fold...")
                print(" Train set smaller than k")
                continue
            else:
                successFold += 1

            # prepare the data in the right format for kNN
            nSamplesTrainData, nxTrain, nyTrain = trainData.shape
            trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
            nSamplesTestData, nxTest, nyTest = testData.shape
            testData = testData.reshape((nSamplesTestData, nxTest * nyTest))

            try:
                res, predictLabels = flast.classifyFlakyTestExperiment(kNN, trainData, testData, trainLabels, testLabels, threshold)
            except:
                print("Skipping fold...")
                print(" Warning on f-measure or conf_matrix", sum(trainLabels))
                continue
            accFlakyCause, info = flast.classifyFlakyCauseExperiment(kNN, testData, trainLabels, predictLabels, trainInfo, testInfo)
            if accFlakyCause != "-":
                res["A_FC"] = accFlakyCause
            res["numFlakyTrainSet"] = sum(trainLabels)
            res["numNonFlakyTrainSet"] = len(trainLabels) - sum(trainLabels)
            res["numFlakyTestSet"] = sum(testLabels)
            res["numNonFlakyTestSet"] = len(testLabels) - sum(testLabels)

            for k, v in res.items():
                resListDict[k].append(v)
            flast.printDict("CLASSIFICATION STATS", res)
            for k, v in info.items():
                resListInfo[k].append(v)
            print(info)
            print()

        if len(resListDict) == 0:
            continue

        outputFile = "{}/{}.csv".format(resultsPath, projectName.replace("/", "_"))

        infoRes = defaultdict(float)
        for k, v in resListInfo.items():
            infoRes[k] = str(sum(v) / len(v))
        infoRes.pop("-", None)

        try:
            FCAcc = sum(resListDict["A_FC"]) / len(resListDict["A_FC"])
        except:
            FCAcc = "-"
        with open(outputFile, "w") as fileOut:
            fileOut.write("A_FC,")
            fileOut.write(",".join(infoRes.keys()))
            fileOut.write("\n")
            fileOut.write("{},".format(FCAcc))
            fileOut.write(",".join(infoRes.values()))
            fileOut.write("\n")

        p1 = time.time()
        print("Running time for project:", p1 - p0)
