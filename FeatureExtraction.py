import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


class FeatureExtraction:
    def __init__(self, pattern="grey-box"):
        self.PATTERN = pattern

    def saliency_map(self, model, sequence, plot=False, path=None):
        if model.name.__contains__("bert"):
            input_ids = sequence['input_ids'].numpy()
            attention_mask = sequence['attention_mask'].numpy()
            token_type_ids = sequence['token_type_ids'].numpy()
            length = input_ids.shape[1]
            input_ids_batch = np.kron(np.ones((length, 1), dtype=np.int32), input_ids)
            attention_mask_batch = np.kron(np.ones((length, 1), dtype=np.int32), attention_mask)
            token_type_ids_batch = np.kron(np.ones((length, 1), dtype=np.int32), token_type_ids)

            for i in range(length):
                input_ids_batch[i, i] = 0
            sequence_batch = sequence.copy()
            sequence_batch['input_ids'] = tf.convert_to_tensor(input_ids_batch)
            sequence_batch['attention_mask'] = tf.convert_to_tensor(attention_mask_batch)
            sequence_batch['token_type_ids'] = tf.convert_to_tensor(token_type_ids_batch)

            prediction = model.predict(sequence)[0]
            label = np.argmax(prediction)
            logit = np.max(prediction)

            predictions = model.predict(sequence_batch)[0]
            logits = predictions[:, label]

            saliency_map = logits - logit

        else:
            length = sequence.shape[0]
            # if self.PATTERN == "grey-box":
            sequence_batch = np.kron(np.ones((length, 1)), sequence)
            for i in range(length):
                sequence_batch[i, i] = 0
            logits = model.predict(sequence_batch)
            logit = model.predict(np.expand_dims(sequence, axis=0))
            saliency_map = logits - logit

        if plot:
            x = int(np.sqrt(length))  # for better presentation of saliency
            np.savetxt(path+".txt", saliency_map)
            # plt.imsave(path+".png", saliency_map.reshape([x, x]))

            # https://matplotlib.org/3.1.0/tutorials/introductory/usage.html
            # import matplotlib
            # matplotlib.use("Agg")
            # plt.imshow(saliency_map.reshape([8, 16]), cmap='viridis', interpolation='nearest')
            plt.imshow(saliency_map.reshape([x, x]), cmap='viridis', interpolation='nearest')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(path+".png", bbox_inches='tight', pad_inches=0, dpi=100)

        return saliency_map

    def saliency_partition(self, model, sequence, num_partition=5, dataset_class=None):
        saliency_map = self.saliency_map(model, sequence)

        if model.name.__contains__("bert"):
            input_ids = sequence['input_ids'].numpy()
            attention_mask = sequence['attention_mask'].numpy()
            token_type_ids = sequence['token_type_ids'].numpy()

            length = input_ids.shape[1]
            valid_index = np.argmax(input_ids == 0)
            if not valid_index:
                valid_index = length
            index = np.expand_dims(np.arange(length), axis=1)
            saliency_absolute = np.concatenate((index, np.abs(np.expand_dims(saliency_map, axis=1))), axis=1)
            saliency_valid = saliency_absolute[0:valid_index]
            saliency_sorted = saliency_valid[saliency_valid[:, 1].argsort()[::-1]]

            partitions = {}
            quotient, remainder = divmod(valid_index, num_partition)
            for key in range(num_partition):
                partitions[key] = [int(saliency_sorted[idx, 0]) for idx in
                                   range(key * quotient, (key + 1) * quotient)]
                if key == num_partition - 1:
                    partitions[key].extend(int(saliency_sorted[idx, 0]) for idx in
                                           range((key + 1) * quotient, valid_index))
            return partitions

        length = sequence.shape[0]
        # be careful about padding location
        valid_index = np.argmax(sequence > 0)

        index = np.expand_dims(np.arange(length), axis=1)
        saliency_absolute = np.concatenate((index, np.abs(saliency_map)), axis=1)
        saliency_valid = saliency_absolute[valid_index:length]
        saliency_sorted = saliency_valid[saliency_valid[:, 1].argsort()[::-1]]

        if dataset_class.DATASET == "imdb":
            partitions = {}
            quotient, remainder = divmod(length - valid_index, num_partition)
            for key in range(num_partition):
                partitions[key] = [int(saliency_sorted[idx, 0]) for idx in
                                   range(key * quotient, (key + 1) * quotient)
                                   if dataset_class.get_word(sequence[valid_index+idx]) != "#"
                                   and dataset_class.get_word(sequence[valid_index+idx]) != "br"]
                if key == num_partition - 1:
                    partitions[key].extend(int(saliency_sorted[idx, 0]) for idx in
                                           range((key + 1) * quotient, length - valid_index)
                                           if dataset_class.get_word(sequence[valid_index+idx]) != "#"
                                           and dataset_class.get_word(sequence[valid_index+idx]) != "br")
            return partitions

        partitions = {}
        quotient, remainder = divmod(length - valid_index, num_partition)
        for key in range(num_partition):
            partitions[key] = [int(saliency_sorted[idx, 0]) for idx in
                               range(key * quotient, (key + 1) * quotient)]
            if key == num_partition - 1:
                partitions[key].extend(int(saliency_sorted[idx, 0]) for idx in
                                       range((key + 1) * quotient, length - valid_index))
        return partitions
