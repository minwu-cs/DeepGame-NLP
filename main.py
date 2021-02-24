from __future__ import print_function

from NeuralNetwork import *
from FeatureExtraction import *
from DataSet import *
from LowerBound import *


# dataSet = "imdb"
# dataSet = "reuters"
dataSet = "sst"
# dataSet = "sst-bert"
# dataSet = "mrpc"

neural_network = NeuralNetwork(dataset=dataSet)
neural_network.train_network(model_type="cnn")
# neural_network.load_network(model_type="bert")
neural_network.MODEL.summary()

# exit()

data_set = DataSet(dataset=dataSet)
feature_extraction = FeatureExtraction()

directory = dataSet + "/"
# directory = dataSet + "/imdb(lstm)_90/"
# directory = "lstm/"
assure_path_exists(directory)

# for index in [9, 41, 94]:
# for index in [73, 81, 170, 253, 272, 297]:
# for index in [6]:
for index in range(100):
    print("index:", index)
    inputSequence = data_set.get_input(index=index)
    text = data_set.get_text(input_ids=inputSequence['input_ids'][0] if neural_network.MODEL.name.__contains__("bert") else inputSequence)
    data_set.print_text(input_ids=inputSequence['input_ids'][0] if neural_network.MODEL.name.__contains__("bert") else inputSequence, index=index)
    oracle = data_set.get_oracle(index=index)
#     # label = neural_network.MODEL.predict_classes(np.expand_dims(inputSequence, axis=0))[0, 0]
#     # label = (neural_network.MODEL.predict(np.expand_dims(inputSequence, axis=0)) > 0.5).astype("int32")[0, 0]
    label, confidence = neural_network.predict(inputSequence)
    print("oracle:", oracle)
    print("predicted label:", label)
    print("predicted confidence:", confidence)
    if text.__contains__("<OOV>"):
        continue

    file_name = "{}-index{}-{}-predict{}{}-{}".format(dataSet,
                                                      index,
                                                      neural_network.MODEL.name,
                                                      label,
                                                      oracle,
                                                      confidence)
    path = directory + file_name
    saliencyMap = feature_extraction.saliency_map(model=neural_network.MODEL,
                                                  sequence=inputSequence,
                                                  plot=True,
                                                  path=path)

    lower_bound = LowerBound(dataset=dataSet,
                             index=index,
                             sequence=inputSequence,
                             model=neural_network,
                             dataset_class=data_set)
    lower_bound.play_game()
