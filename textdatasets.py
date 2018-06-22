from annoy import AnnoyIndex
import statistics
from names_cleanser import NameDataCleanser
from torch.utils.data import Dataset
import torch
from embeddings import KazumaCharEmbedding
import numpy as np
import random

TRAIN_NEIGHBOR_LEN = 20
TEST_NEIGHBOR_LEN = 20
EMBEDDING_DIM = 100
MAX_NB_WORDS = 140000
MAX_SEQUENCE_LENGTH = 10

DEBUG = True
DEBUG_DATA_LENGTH = 100

class ANNBasedTripletSelector:

    def get_embeddings(self):
        num_words = len(self.word2idx)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        print('about to get kz')
        kz = KazumaCharEmbedding()
        print('got kz')

        for word, i in self.word2idx.items():
            if i >= MAX_NB_WORDS:
                continue
            embedding_vector = kz.emb(word)
            if embedding_vector is not None:
                if sum(embedding_vector) == 0:
                    print("failed to find embedding for:" + word)
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        self.idx_to_embedding = embedding_matrix


    def build_word_to_idx(self):
        unique_words = set()
        self.word2idx = {}

        for entity in self.entities:
            for word in entity.split():
                unique_words.add(word)

        i = 0
        for word in unique_words:
            self.word2idx[word] = i
            i += 1


    def convert_entity_to_sequence(self, entity):
        words = entity.split()

        sequence = []
        maxlen = min(len(words), MAX_SEQUENCE_LENGTH)

        for i in range(0, maxlen):
            sequence.append(self.idx_to_embedding[self.word2idx[words[i]]])

        if len(words) < MAX_SEQUENCE_LENGTH:
            for i in range(len(words), MAX_SEQUENCE_LENGTH):
                sequence.append(np.zeros(EMBEDDING_DIM))

        return torch.from_numpy(np.asarray(sequence))

    def get_entities_as_sequence(self, ents):
        sequences = []
        for e in ents:
            print(e)
            sequences.append(self.convert_entity_to_sequence(e))
        return sequences

    def get_tensors(self, triples):
        ret = {}
        for text_type in triples:
            len(triples[text_type])
            lst = list(map(lambda t: self.convert_entity_to_sequence(t), triples[text_type]))
            ret[text_type] = lst
            assert len(lst) == len(triples[text_type])
        return ret

    @staticmethod
    def generate_triplets_from_ann(embeddings, entity2unique, entity2same, unique_text, test):
        t = AnnoyIndex(len(embeddings[0].numpy().flatten()), metric='euclidean')  # Length of item vector that will be indexed
        t.set_seed(123)
        for i in range(len(embeddings)):
            v = embeddings[i].numpy().flatten()
            t.add_item(i, v)

        t.build(100)  # 100 trees

        match = 0
        no_match = 0
        accuracy = 0
        total = 0

        triplets = {}

        pos_distances = []
        neg_distances = []

        triplets['anchor'] = []
        triplets['positive'] = []
        triplets['negative'] = []

        if test:
            NNlen = TEST_NEIGHBOR_LEN
        else:
            NNlen = TRAIN_NEIGHBOR_LEN

        for key in entity2same:
            index = entity2unique[key]
            nearest = t.get_nns_by_vector(embeddings[index].numpy().flatten(), NNlen)
            nearest_text = set([unique_text[i] for i in nearest])
            expected_text = set(entity2same[key])
            # annoy has this annoying habit of returning the queried item back as a nearest neighbor.  Remove it.
            if key in nearest_text:
                nearest_text.remove(key)
            # print("query={} names = {} true_match = {}".format(unique_text[index], nearest_text, expected_text))
            overlap = expected_text.intersection(nearest_text)
            # collect up some statistics on how well we did on the match
            m = len(overlap)
            match += m
            # since we asked for only x nearest neighbors, and we get at most x-1 neighbors that are not the same as
            # key (!) make sure we adjust our estimate of no match appropriately
            no_match += min(len(expected_text), NNlen - 1) - m

            # sample only the negatives that are true negatives
            # that is, they are not in the expected set - sampling only 'semi-hard negatives is not defined here'
            # positives = expected_text - nearest_text
            positives = expected_text
            negatives = nearest_text - expected_text

            # print(key + str(expected_text) + str(nearest_text))
            for i in negatives:
                for j in positives:
                    dist_pos = t.get_distance(index, entity2unique[j])
                    pos_distances.append(dist_pos)
                    dist_neg = t.get_distance(index, entity2unique[i])
                    neg_distances.append(dist_neg)
                    if dist_pos < dist_neg:
                        accuracy += 1
                    total += 1
                    # print(key + "|" +  j + "|" + i)
                    # print(dist_pos)
                    # print(dist_neg)
                    triplets['anchor'].append(key)
                    triplets['positive'].append(j)
                    triplets['negative'].append(i)

        print("mean positive distance:" + str(statistics.mean(pos_distances)))
        print("stdev positive distance:" + str(statistics.stdev(pos_distances)))
        print("max positive distance:" + str(max(pos_distances)))
        print("mean neg distance:" + str(statistics.mean(neg_distances)))
        print("stdev neg distance:" + str(statistics.stdev(neg_distances)))
        print("max neg distance:" + str(max(neg_distances)))
        print("Accuracy in the ANN for triplets that obey the distance func:" + str(accuracy / total))

        if test:
            return match / (match + no_match)
        else:
            return triplets, match / (match + no_match)

    @staticmethod
    def read_entities(filepath):
        lineindex = 0
        ents = []
        with open(filepath) as fl:
            for line in fl:
                lineindex += 1
                if DEBUG and lineindex > DEBUG_DATA_LENGTH:
                    continue
                else:
                    ents.append(line)
        return ents

    @staticmethod
    def split(ents, test_split=0.2):
        if not DEBUG:
            random.shuffle(ents)
        num_validation_samples = int(test_split * len(ents))
        return ents[:-num_validation_samples], ents[-num_validation_samples:]

    @staticmethod
    def build_unique_entities(entity2same):
        unique_text = []
        entity2index = {}

        for key in entity2same:
            entity2index[key] = len(unique_text)
            unique_text.append(key)
            values = entity2same[key]
            for v in values:
                entity2index[v] = len(unique_text)
                unique_text.append(v)

        return unique_text, entity2index

    @staticmethod
    def generate_names(entities, limit_pairs=False):
        num_names = 4
        names_generator = NameDataCleanser(0, num_names, limit_pairs=limit_pairs)
        entity2same = {}
        for entity in entities:
            ret = names_generator.cleanse_data(entity)
            if ret and len(ret) >= num_names:
                entity2same[ret[0]] = ret[1:]
        return entity2same

    def get_all_entities(self, d):
        for k, v in d.items():
            self.entities.append(k)
            self.entities.extend(v)

    def __init__(self, filename):
        self.entities = []
        ents = self.read_entities(filename)
        self.train, self.test = self.split(ents)
        entity2same_train = self.generate_names(self.train)
        entity2same_test = self.generate_names(self.test, limit_pairs=True)
        print(len(entity2same_train))
        print(len(entity2same_test))

        self.get_all_entities(entity2same_train)
        self.get_all_entities(entity2same_test)

        self.word2idx = {}
        self.build_word_to_idx()
        print(self.word2idx)
        self.idx_to_embedding = {}
        self.get_embeddings()

        # build a set of data structures useful for annoy, the set of unique entities (unique_text),
        # a mapping of entities in texts to an index in unique_text, a mapping of entities to other same entities, and the actual
        # vectorized representation of the text.  These structures will be used iteratively as we build up the model
        # so we need to create them once for re-use
        unique_text_train, entity2unique_train = self.build_unique_entities(entity2same_train)
        unique_text_test, entity2unique_test = self.build_unique_entities(entity2same_test)

        print("train text len:" + str(len(unique_text_train)))
        print("test text len:" + str(len(unique_text_test)))

        test_seq = self.get_entities_as_sequence(unique_text_test)
        test_data, test_match_stats = self.generate_triplets_from_ann(test_seq, entity2unique_test, entity2same_test, unique_text_test, False)
        self.test_data = self.get_tensors(test_data)
        print("Test stats:" + str(test_match_stats))

        train_seq = self.get_entities_as_sequence(unique_text_train)
        train_data, match_stats = self.generate_triplets_from_ann(train_seq, entity2unique_train, entity2same_train, unique_text_train, False)
        self.train_data = self.get_tensors(train_data)
        print("Match stats:" + str(match_stats))

    def get_test_data(self):
        return self.test_data

    def get_train_data(self):
        return self.train_data

class TripletDataset(Dataset):
    def __init__(self, triplet_selector, train):
        self.dataset = triplet_selector
        self.train = train
        if train:
            self.train = triplet_selector.get_train_data()
        else:
            self.test = triplet_selector.get_test_data()

    def __getitem__(self, index):

        if self.train:
            t = self.train
        else:
            t = self.test
        return (t['anchor'][index], t['positive'][index], t['negative'][index]), []

    def __len__(self):
        if self.train:
            t = self.train
        else:
            t = self.test
        return len(t)
