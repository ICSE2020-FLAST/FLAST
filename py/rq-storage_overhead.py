from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import sparse
import numpy as np
import os
import pickle
import time

import flast


if __name__ == "__main__":
    print("RQ - STORAGE OVERHEAD")

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
        "DeFlaker/handlebars",  # 7 vs 523
        "DeFlaker/logback",  # 11 vs 1041
        "DeFlaker/okhttp",  # 32 vs 1199
        "DeFlaker/tachyon",  # 7 vs 609
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
    numSplits = 10  # number of splits for StratifiedShuffleSplit Cross Validation

    # DO NOT TOUCH THIS
    # parameters transformation from FLAST parameters to parameters for experiments
    if threshold < 0.5 or threshold > 1:
        print("FLAST threshold must be in [0.5, 1]")
        exit()
    thresholdName = "t-0.5" if threshold == 0 else "t-{}".format(threshold)
    if threshold == 0.5:
        threshold = 0
    reduceDim = False if dim < 0 else True

    resultsPath = "results/rq-storage_overhead/{}".format(thresholdName)
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/rq-storage_overhead/"):
        os.makedirs("results/rq-storage_overhead")
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    outputFile = "{}/rq-storage_overhead.csv".format(resultsPath)
    with open(outputFile, "w") as fileOut:
        fileOut.write("Dataset,Storage Overhead\n")

    for projectName in projectList:
        p0 = time.time()
        print()
        print("#"*80)
        print("TESTING {}".format(projectName.upper()))

        # FLAST setup
        kNN = KNeighborsClassifier(
            algorithm=kNNParams["algorithm"],
            metric=kNNParams["metric"],
            n_neighbors=kNNParams["n_neighbors"],
            weights=kNNParams["weights"],
            n_jobs=kNNParams["n_jobs"])

        dataPointsFlaky, dataPointsNonFlaky, flakyInfoList = flast.getDataPointsInfo(projectBasePath, projectName)

        print(" Project Name:", projectName)
        print(" Number of Flaky Tests:", len(dataPointsFlaky))
        print(" Total Number of Tests:", len(dataPointsFlaky) + len(dataPointsNonFlaky))

        # create dataPointsList
        dataPoints = dataPointsFlaky + dataPointsNonFlaky
        Z = flast.vectorizeDataPointsSRP(dataPoints, reduceDim=reduceDim, dim=dim, eps=eps)

        dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
        dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))

        # test_size = 0.1 is the same size used for CV in the experiments
        trainData, testData, trainLabels, testLabels = train_test_split(dataPointsList, dataLabelsList, test_size=0.1, random_state=42)
        # prepare the data in the right format for kNN
        nSamplesTrainData, nxTrain, nyTrain = trainData.shape
        trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))

        kNN = (sparse.coo_matrix(trainData), sparse.coo_matrix(trainLabels))
        pickleDumpKNN = "{}/{}.pickle".format(resultsPath, projectName.replace("/", "_"))
        # pickle.dump(kNN, open(pickleDumpKNN, "wb"))
        storage = len(pickle.dumps(kNN, -1))
        print("Storage overhead:", storage)

        with open(outputFile, "a") as fileOut:
            fileOut.write("{},{}\n".format(projectName, storage))

        p1 = time.time()
        print("Running time for project:", p1 - p0)
