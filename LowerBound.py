from typing import List, Any, Union
import heapq

from FeatureExtraction import *
from DataSet import *
from basics import *


class LowerBound:
    def __init__(self, dataset, index, sequence, model, dataset_class):
        self.DATASET = dataset
        self.INDEX = index
        self.SEQUENCE = sequence
        self.MODEL = model
        self.LABEL, self.CONFIDENCE = self.MODEL.predict(self.SEQUENCE)

        #
        # self.DATASET_CLASS = DataSet(dataset=dataset)
        self.DATASET_CLASS = dataset_class
        #

        self.MANIPULATION = "substitution"
        if self.MANIPULATION == "substitution":
            # text = self.DATASET_CLASS.get_text(input_ids=sequence)
            self.NEIGHBOURHOOD = self.DATASET_CLASS.get_neighbourhood(input_ids=sequence)
        self.DIST_METRIC = "WMD"
        self.DIST_BUDGET = 50
        self.TAU = 0.001

        feature_extraction = FeatureExtraction(pattern="grey-box")
        self.PARTITIONS = feature_extraction.saliency_partition(model=self.MODEL.MODEL, sequence=self.SEQUENCE, dataset_class=dataset_class)

        self.DIST_ESTIMATION = {}
        self.ADV_MANIPULATION = ()
        self.ADVERSARY_FOUND = None
        self.ADVERSARY = None
        self.CURRENT_SAFE = []

        self.NUM_ITERATION = 0
        self.MAX_ITERATION = 1000

        assure_path_exists("%s/" % self.DATASET)

    def compute_distance(self, sequence_x, sequence_y, embedding="keras"):
        if self.DIST_METRIC == 'WMD':
            # if self.MODEL.MODEL.name.__contains__("bert"):
            #     sequence_x = sequence_x['input_ids'][0]
            #     sequence_y = sequence_x['input_ids'][0]
            sequence_x = self.DATASET_CLASS.get_text(input_ids=sequence_x)
            sequence_y = self.DATASET_CLASS.get_text(input_ids=sequence_y)
            return word_movers_distance(sequence_x, sequence_y)
        elif embedding == "keras":
            sequence_x = self.MODEL.obtain_embedding(sequence_x)
            sequence_y = self.MODEL.obtain_embedding(sequence_y)

        if self.DIST_METRIC == "L2":
            return l2_distance(sequence_x, sequence_y)
        elif self.DIST_METRIC == "L1":
            return l1_distance(sequence_x, sequence_y)
        elif self.DIST_METRIC == "Linf":
            return linf_distance(sequence_x, sequence_y)

    # @staticmethod
    def apply_manipulation(self, sequence, locations, pattern="deletion"):
        if pattern == "deletion":
            if self.MODEL.MODEL.name.__contains__("bert"):
                input_ids = sequence['input_ids'].numpy()
                attention_mask = sequence['attention_mask'].numpy()
                token_type_ids = sequence['token_type_ids'].numpy()
                # length = input_ids.shape[1]
                length = len(locations)
                input_ids_batch = np.kron(np.ones((length, 1), dtype=np.int32), input_ids)
                attention_mask_batch = np.kron(np.ones((length, 1), dtype=np.int32), attention_mask)
                token_type_ids_batch = np.kron(np.ones((length, 1), dtype=np.int32), token_type_ids)
                atomic_list = []
                for idx in range(length):
                    input_ids_batch[idx, locations[idx]] = 0
                    atomic_list.append((locations[idx], 0))
                sequence_batch = sequence.copy()
                sequence_batch['input_ids'] = tf.convert_to_tensor(input_ids_batch)
                sequence_batch['attention_mask'] = tf.convert_to_tensor(attention_mask_batch)
                sequence_batch['token_type_ids'] = tf.convert_to_tensor(token_type_ids_batch)
                return sequence_batch, atomic_list

            length = len(locations)
            atomic_list = []
            sequence_batch = np.kron(np.ones((length, 1)), sequence)
            for idx in range(length):
                sequence_batch[idx, locations[idx]] = 0
                atomic_list.append((locations[idx], 0))
            return sequence_batch, atomic_list
        elif pattern == "substitution":
            atomic_list = []
            sequence_batch = []
            for location in locations:
                substitutions = self.NEIGHBOURHOOD.get(location)
                if substitutions:
                    for substitution in substitutions:
                        atomic_list.append((location, substitution))
            if atomic_list:
                length = len(atomic_list)
                sequence_batch = np.kron(np.ones((length, 1)), sequence)
                for idx in range(length):
                    sequence_batch[idx, atomic_list[idx][0]] = atomic_list[idx][1]
            return sequence_batch, atomic_list
            # print("Word substitution.")
        else:
            print("Unrecognised manipulation pattern. "
                  "Try 'deletion' or 'substitution'.")

    def target_locations(self, sequence, locations):
        manipulated_sequences, atomic_list = self.apply_manipulation(sequence, locations, pattern=self.MANIPULATION)
        # print(len(manipulated_sequences))
        # print(len(atomic_list))
        if not atomic_list:
            return

        if self.MODEL.MODEL.name.__contains__("bert"):
            logits = self.MODEL.MODEL.predict(manipulated_sequences)[0]
            input_ids = sequence['input_ids'][0]
            input_ids_batch = manipulated_sequences['input_ids']
            for idx in range(len(atomic_list)):
                if (input_ids_batch[idx].numpy() == input_ids.numpy()).all():
                    continue
                cost = self.compute_distance(input_ids_batch[idx], self.SEQUENCE['input_ids'][0])
                [logit_max, logit_2dn_max] = heapq.nlargest(2, logits[idx])
                heuristic = (logit_max - logit_2dn_max) * 2 * self.TAU  # heuristic value
                estimation = cost + heuristic

                atomic_manipulation = atomic_list[idx]

                valid = True
                if self.MANIPULATION == "substitution" and self.ADV_MANIPULATION:
                    atomics = [self.ADV_MANIPULATION[i:i + 2] for i in range(0, len(self.ADV_MANIPULATION), 2)]
                    for atomic in atomics:
                        if atomic[0] == atomic_manipulation[0]:
                            valid = False
                            break
                if not valid:
                    continue
                current = self.ADV_MANIPULATION + atomic_manipulation
                current_lst = [current[i:i + 2] for i in range(0, len(current), 2)]
                # valid = True
                for key in self.DIST_ESTIMATION.keys():
                    key_list = [key[i:i + 2] for i in range(0, len(key), 2)]
                    if set(key_list) == set(current_lst):
                        valid = False
                        # print("Already added.")
                        break
                # self.DIST_ESTIMATION.update({self.ADV_MANIPULATION + atomic_manipulation: estimation})
                if valid:
                    self.DIST_ESTIMATION.update({current: estimation})

        else:
            # logits = self.MODEL.predict(manipulated_sequences)
            logits = self.MODEL.MODEL.predict(manipulated_sequences)

            for idx in range(len(manipulated_sequences)):
                if (manipulated_sequences[idx] == sequence).all():
                    continue
                cost = self.compute_distance(manipulated_sequences[idx], self.SEQUENCE)
                heuristic = np.abs(logits[idx] - (1-logits[idx])) * 2 * self.TAU
                # heuristic = (logits[idx] - self.CONFIDENCE) * 2 * self.TAU
                estimation = cost + heuristic

                # atomic_manipulation = (locations[idx], 0)
                atomic_manipulation = atomic_list[idx]

                valid = True
                if self.MANIPULATION == "substitution" and self.ADV_MANIPULATION:
                    atomics = [self.ADV_MANIPULATION[i:i + 2] for i in range(0, len(self.ADV_MANIPULATION), 2)]
                    for atomic in atomics:
                        if atomic[0] == atomic_manipulation[0]:
                            valid = False
                            break
                if not valid:
                    continue
                current = self.ADV_MANIPULATION + atomic_manipulation
                current_lst = [current[i:i + 2] for i in range(0, len(current), 2)]
                # valid = True
                for key in self.DIST_ESTIMATION.keys():
                    key_list = [key[i:i + 2] for i in range(0, len(key), 2)]
                    if set(key_list) == set(current_lst):
                        valid = False
                        # print("Already added.")
                        break
                # self.DIST_ESTIMATION.update({self.ADV_MANIPULATION + atomic_manipulation: estimation})
                if valid:
                    self.DIST_ESTIMATION.update({current: estimation})

        # print("Atomic manipulations of target locations done.")

    def play_game(self):

        if self.MODEL.MODEL.name.__contains__("bert"):
            new_sequence = self.SEQUENCE.copy()
            new_label, new_confidence = self.MODEL.predict(new_sequence)
            # dist = self.compute_distance(self.SEQUENCE, new_sequence)
            dist = self.compute_distance(sequence_x=self.SEQUENCE['input_ids'][0] if self.MODEL.MODEL.name.__contains__("bert") else self.SEQUENCE,
                                         sequence_y=new_sequence['input_ids'][0] if self.MODEL.MODEL.name.__contains__("bert") else new_sequence)

            while dist <= self.DIST_BUDGET and new_label == self.LABEL:
                for partitionID in self.PARTITIONS.keys():
                    locations = self.PARTITIONS[partitionID]
                    self.target_locations(sequence=new_sequence, locations=locations)

                if not self.DIST_ESTIMATION:
                    print("Manipulations finished.")
                    print("Lower bounds:")
                    print(self.CURRENT_SAFE)
                    path = "{}/{}-index{}-{}-lb-neighbour{}".format(self.DATASET,
                                                                    self.DATASET,
                                                                    self.INDEX,
                                                                    self.MODEL.MODEL.name,
                                                                    self.DATASET_CLASS.SIMILARITY_BOUND)
                    # path = "%s/%s-index%d-%s-lb" % (self.DATASET, self.DATASET, self.INDEX, self.MODEL.MODEL.name)
                    np.savetxt(path + ".txt", self.CURRENT_SAFE)
                    break

                self.ADV_MANIPULATION = min(self.DIST_ESTIMATION, key=self.DIST_ESTIMATION.get)
                print("Current best manipulations:", self.ADV_MANIPULATION)
                # print("%s distance (estimated): %s" % (self.DIST_METRIC, self.DIST_ESTIMATION[self.ADV_MANIPULATION][0]))
                self.DIST_ESTIMATION.pop(self.ADV_MANIPULATION)

                new_sequence = self.SEQUENCE.copy()
                atomic_list = [self.ADV_MANIPULATION[i:i + 2] for i in range(0, len(self.ADV_MANIPULATION), 2)]
                if self.MODEL.MODEL.name.__contains__("bert"):
                    manipulated_words = [self.DATASET_CLASS.get_word(input_id=new_sequence['input_ids'][:,atomic[0]]) for atomic in atomic_list]
                else:
                    manipulated_words = [self.DATASET_CLASS.get_word(new_sequence[atomic[0]]) for atomic in atomic_list]
                print('Manipulated word(s):', manipulated_words)

                temp = new_sequence['input_ids'].numpy()
                for atomic in atomic_list:
                    # for atomic in self.ADV_MANIPULATION:
                    # new_sequence[atomic[0]] = atomic[1]
                    temp[0, atomic[0]] = atomic[1]
                new_sequence['input_ids'] = tf.convert_to_tensor(temp)

                # dist = self.compute_distance(self.SEQUENCE, new_sequence)
                dist = self.compute_distance(
                    sequence_x=self.SEQUENCE['input_ids'][0] if self.MODEL.MODEL.name.__contains__(
                        "bert") else self.SEQUENCE,
                    sequence_y=new_sequence['input_ids'][0] if self.MODEL.MODEL.name.__contains__(
                        "bert") else new_sequence)
                print("%s distance (actual): %s" % (self.DIST_METRIC, dist))

                self.CURRENT_SAFE.append(dist)

                new_label, new_confidence = self.MODEL.predict(new_sequence)
                if dist > self.DIST_BUDGET:
                    print("Adversarial distance exceeds distance budget.")
                    self.ADVERSARY_FOUND = False
                    break
                elif new_label != self.LABEL:
                    print("Adversarial sequence is found.")
                    self.ADVERSARY_FOUND = True
                    self.ADVERSARY = new_sequence
                    print("Lower bounds:")
                    print(self.CURRENT_SAFE)
                    path = "{}/{}-index{}-{}-lb-neighbour{}".format(self.DATASET,
                                                                    self.DATASET,
                                                                    self.INDEX,
                                                                    self.MODEL.MODEL.name,
                                                                    self.DATASET_CLASS.SIMILARITY_BOUND)
                    # path = "%s/%s-index%d-%s-lb" % (self.DATASET, self.DATASET, self.INDEX, self.MODEL.MODEL.name)
                    np.savetxt(path + ".txt", self.CURRENT_SAFE)
                    break

                # if self.CURRENT_SAFE[-1] != dist:
                #     path = "%s_lb/idx_%s_Safe_currentBest_%d.txt" % (self.DATASET, self.INDEX, dist)
                #     np.savetxt(path, new_sequence)

                # self.CURRENT_SAFE.append(dist)

                self.NUM_ITERATION = self.NUM_ITERATION + 1
                if self.NUM_ITERATION == self.MAX_ITERATION:
                    print("Iteration reaches %d." % self.NUM_ITERATION)
                    print("Lower bounds:")
                    print(self.CURRENT_SAFE)
                    path = "{}/{}-index{}-{}-lb-neighbour{}".format(self.DATASET,
                                                                    self.DATASET,
                                                                    self.INDEX,
                                                                    self.MODEL.MODEL.name,
                                                                    self.DATASET_CLASS.SIMILARITY_BOUND)
                    # path = "%s/%s-index%d-%s-lb" % (self.DATASET, self.DATASET, self.INDEX, self.MODEL.MODEL.name)
                    np.savetxt(path + ".txt", self.CURRENT_SAFE)
                    break

        else:
            new_sequence = self.SEQUENCE.copy()
            new_label, new_confidence = self.MODEL.predict(new_sequence)
            dist = self.compute_distance(self.SEQUENCE, new_sequence)

            while dist <= self.DIST_BUDGET and new_label == self.LABEL:
                for partitionID in self.PARTITIONS.keys():
                    locations = self.PARTITIONS[partitionID]
                    self.target_locations(sequence=new_sequence, locations=locations)

                if not self.DIST_ESTIMATION:
                    print("Manipulations finished.")
                    print("Lower bounds:")
                    print(self.CURRENT_SAFE)
                    path = "{}/{}-index{}-{}-lb-neighbour{}".format(self.DATASET,
                                                                    self.DATASET,
                                                                    self.INDEX,
                                                                    self.MODEL.MODEL.name,
                                                                    self.DATASET_CLASS.SIMILARITY_BOUND)
                    # path = "%s/%s-index%d-%s-lb" % (self.DATASET, self.DATASET, self.INDEX, self.MODEL.MODEL.name)
                    np.savetxt(path + ".txt", self.CURRENT_SAFE)
                    break

                self.ADV_MANIPULATION = min(self.DIST_ESTIMATION, key=self.DIST_ESTIMATION.get)
                print("Current best manipulations:", self.ADV_MANIPULATION)
                # print("%s distance (estimated): %s" % (self.DIST_METRIC, self.DIST_ESTIMATION[self.ADV_MANIPULATION][0]))
                self.DIST_ESTIMATION.pop(self.ADV_MANIPULATION)

                new_sequence = self.SEQUENCE.copy()
                atomic_list = [self.ADV_MANIPULATION[i:i+2] for i in range(0, len(self.ADV_MANIPULATION), 2)]
                manipulated_words = [self.DATASET_CLASS.get_word(new_sequence[atomic[0]]) for atomic in atomic_list]
                print('Manipulated word(s):', manipulated_words)
                for atomic in atomic_list:
                # for atomic in self.ADV_MANIPULATION:
                    new_sequence[atomic[0]] = atomic[1]
                dist = self.compute_distance(self.SEQUENCE, new_sequence)
                print("%s distance (actual): %s" % (self.DIST_METRIC, dist))

                self.CURRENT_SAFE.append(dist)

                new_label, new_confidence = self.MODEL.predict(new_sequence)
                if dist > self.DIST_BUDGET:
                    print("Adversarial distance exceeds distance budget.")
                    self.ADVERSARY_FOUND = False
                    break
                elif new_label != self.LABEL:
                    print("Adversarial sequence is found.")
                    self.ADVERSARY_FOUND = True
                    self.ADVERSARY = new_sequence
                    print("Lower bounds:")
                    print(self.CURRENT_SAFE)
                    path = "{}/{}-index{}-{}-lb-neighbour{}".format(self.DATASET,
                                                                    self.DATASET,
                                                                    self.INDEX,
                                                                    self.MODEL.MODEL.name,
                                                                    self.DATASET_CLASS.SIMILARITY_BOUND)
                    # path = "%s/%s-index%d-%s-lb" % (self.DATASET, self.DATASET, self.INDEX, self.MODEL.MODEL.name)
                    np.savetxt(path + ".txt", self.CURRENT_SAFE)
                    break

                # if self.CURRENT_SAFE[-1] != dist:
                #     path = "%s_lb/idx_%s_Safe_currentBest_%d.txt" % (self.DATASET, self.INDEX, dist)
                #     np.savetxt(path, new_sequence)

                # self.CURRENT_SAFE.append(dist)

                self.NUM_ITERATION = self.NUM_ITERATION + 1
                if self.NUM_ITERATION == self.MAX_ITERATION:
                    print("Iteration reaches %d." % self.NUM_ITERATION)
                    print("Lower bounds:")
                    print(self.CURRENT_SAFE)
                    path = "{}/{}-index{}-{}-lb-neighbour{}".format(self.DATASET,
                                                                    self.DATASET,
                                                                    self.INDEX,
                                                                    self.MODEL.MODEL.name,
                                                                    self.DATASET_CLASS.SIMILARITY_BOUND)
                    # path = "%s/%s-index%d-%s-lb" % (self.DATASET, self.DATASET, self.INDEX, self.MODEL.MODEL.name)
                    np.savetxt(path + ".txt", self.CURRENT_SAFE)
                    break





