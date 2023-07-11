import torch
import torch.nn.functional as F
import numpy as np
import operator
import functools
import math
import csv
import json
import sys
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from typing import Tuple, Dict, AnyStr
from numbers import Number


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def open_word_list_csv():
    with open("./aoa_data/word_list.csv", newline='') as csv_file:
        word_list = []
        csv_contents = csv.DictReader(csv_file, delimiter=',', quotechar='|')
        for row in csv_contents:
            word_list.append(row)
        return word_list

## Custom Dataset class for CHILDES utterances
class CHILDESDataset(Dataset):
    def __init__(self, file_path):
        self.sentences = []
        self.words = []
        with open(file_path, "r") as f:
            sent_words = json.load(f)
            for item in sent_words:
                self.sentences.append(item['sent'])
                self.words.append(item['words'])
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, index):
        sent = self.sentences[index]
        words = '-'.join(self.words[index])
        return {'sent':sent, 'words': words}

# create mapping from words to tokens
def make_token_word_mappings(word_list_dict, tokenizer):
    word_mappings = dict()
    for word in word_list_dict:
        w = word['word_clean']
        s_w = " "+w
        seq = tokenizer(w, add_special_tokens=False)['input_ids']
        # if tokenizer has different tokens for words at the begining of a string vs followed by a space, get both.
        s_seq = tokenizer(s_w, add_special_tokens=False)['input_ids']
        if seq == s_seq:
            tokens= [torch.Tensor(seq).long()]
            n_tokens= [len(seq)]
        else:
            tokens= [torch.Tensor(seq).long(), torch.Tensor(s_seq).long()]
            n_tokens= [len(seq), len(s_seq)]
        word_mappings[w] = (tokens, n_tokens)
    return word_mappings

# find index matches for sequences of indexes i.e. if a word is tokenized as multiple tokens
def indexes_in_sequence(query, base):
    batch_id, label = base[0], base[1]
    label = label.squeeze()
    l = len(query)
    locations = []
    for index in range((len(label)-l)):
        if torch.all(label[index:index+l] == query):
            locations.append([batch_id, index])
    return locations

# get average surprisal values from model
r"""
    Note for Causal LMs, the code currently assumes that labels **are shifted** inside the model, like GPT and OPT models,
    i.e. we can set ``labels = input_ids``. If this is not the case, you can either provide the labels using your tokenizer
    under the key ['labels'] which the code currently handles, or you will have to adjust it to make sure that labels shift
    accordingly here.
"""
def get_batched_surprisal(model, tokenizer, model_type, dataloader, word_mapping, device):
    model.eval()
    word_surprisals_n = {}
    for word in word_mapping.keys():
        word_surprisals_n[word] = [0, 0]
    batch_size = dataloader.batch_size
    for n, item in tqdm(enumerate(dataloader), total=len(dataloader)):
        sentences = item['sent']
        words = item['words']
        batch = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=50)
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        if model_type == "hf-causal":
            if 'labels' in batch.keys():
                labels = batch['labels']
            else:
                labels = batch['input_ids']
        else:
            labels = batch['input_ids']
        outputs = model(**batch, labels=labels)
        surprisals = -F.log_softmax(outputs.logits, -1)
        labels_split = torch.tensor_split(labels, batch_size)
        word_list = list(set(sum([w.split('-') for w in words], [])))
        for word in word_list:
            tokens, n_tokens =  word_mapping[word]
            for indexes, n_tokens in zip(tokens, n_tokens):
                indexes = indexes.to(device)
                # If word is represented by a single token we can optimize run time by searching over whole batch matrix
                if n_tokens == 1:
                    index_matches = (labels == indexes).nonzero(as_tuple=False)
                    if len(index_matches) > 0:
                        for i in index_matches:
                            match = surprisals[tuple(i)]
                            surprisal = match[indexes].item()
                            word_surprisals_n[word][0] += surprisal
                            word_surprisals_n[word][1] += 1
                # If word is represented by sequence of tokens, we have to match the whole sequence
                else:
                    match_list = list(map(lambda x: indexes_in_sequence(indexes, x), enumerate(labels_split)))
                    index_matches = functools.reduce(operator.iconcat, match_list)
                    if len(index_matches) > 0:
                        for i in index_matches:
                            surprisal = 0.0
                            batch_id = int(i[0])
                            # we sum the log probs of each sub token in sequence to get sequence surprisal
                            for j, index in enumerate(indexes):
                                current_id = int(i[1] + j)
                                match = surprisals[(batch_id, current_id)]
                                sub_surprisal = match[index].item()
                                surprisal += sub_surprisal
                            word_surprisals_n[word][0] += surprisal
                            word_surprisals_n[word][1] += 1
    # Average the summed suprisal values for each non zero count word
    for word in word_mapping.keys():
        if word_surprisals_n[word][1] > 0:
            word_surprisals_n[word][0] = word_surprisals_n[word][0]/ word_surprisals_n[word][1]
    return word_surprisals_n

# fit linear regressions using LOO cross validation to get MAD scores for predicting AoA
def get_loo_mad_results(word_surprisals_n, word_list_dict):
    data = []
    # collect all predictors in one table
    for w in word_list_dict:
        word = w['word_clean']
        lex_cat = w['lexical_category']
        concreteness = w['concreteness']
        frequency = w['frequency']
        aoa = w['aoa']
        avg_surprisal, count = word_surprisals_n[word]
        if count != 0:
            data.append([word, float(aoa), lex_cat, float(concreteness), avg_surprisal, int(frequency)])
    data = np.array(data)
    # turn frequency counts into unigram surprisals
    total = np.sum(data[:,5].astype(int))
    unigram = -np.log(data[:,[5]].astype(int) / total)
    # get residualized surprisal values
    surp_model = LinearRegression().fit(unigram, data[:,4])
    prediction = surp_model.predict(unigram)
    residuals = (data[:,4].astype(float) - prediction)
    resid_surp = np.expand_dims(residuals, axis=1)
    data = np.concatenate((data[:,0:4], resid_surp, unigram), axis=1)
    # dummy code lex_category with noun as base and predicate, function_words as separate dummy codes
    enc = OneHotEncoder(drop=['nouns'], categories=[['nouns', 'function_words', 'predicates']])
    dummy_coded_categorical_vars = enc.fit_transform(data[:,[2]]).toarray()
    # standardize variables so mean=0 and std=1
    scaler = StandardScaler()
    scaled_continuous_vars = scaler.fit_transform(data[:,3:])
    scaled_data = np.concatenate((dummy_coded_categorical_vars, scaled_continuous_vars), axis=1)
    # get interaction term values between lexical category dummy codes and concreteness/surprisal/frequency add to dataset
    inter = PolynomialFeatures(interaction_only=True, include_bias=False)
    interactions = inter.fit_transform(scaled_data)
    # get LOO data splits
    X = np.concatenate((interactions[:,:5], interactions[:,6:12]), axis=1)
    y = data[:,1]
    loo = LeaveOneOut()
    n = loo.get_n_splits(X)
    overall_mad = 0
    noun_n = 0
    noun_mad = 0
    pred_n = 0
    pred_mad = 0
    fctword_n = 0
    fctword_mad = 0
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        # for each split fit linear regression model and get absolute deviation
        model = LinearRegression().fit(X[train_index], y[train_index])
        y_pred = model.predict(X[test_index])
        abs_deviation = abs(float(y[test_index]) - float(y_pred))
        overall_mad += abs_deviation
        if bool(X[test_index, 1] == 1):
            pred_mad += abs_deviation
            pred_n += 1
        elif bool(X[test_index, 0] == 1):
            fctword_mad += abs_deviation
            fctword_n += 1
        else:
            noun_mad += abs_deviation
            noun_n += 1
    overall_mad = overall_mad/n
    noun_mad = noun_mad/noun_n
    pred_mad = pred_mad/pred_n
    fctword_mad = fctword_mad/fctword_n
    return {'overall_mad':overall_mad, 'n':n, 'noun_mad':noun_mad, 'n_noun':noun_n, 'predicate_mad':pred_mad, 'n_predicate':pred_n, 'functionword_mad':fctword_mad, 'n_functionword':fctword_n}


def aoa_pred_eval(model: object, tokenizer: object, model_type: str, batch_size: int = 32) -> Tuple[Dict[AnyStr, Tuple[Number, Number]], Dict[AnyStr, Number]]:
    logger.info(f"\n» Evaluating model on predicting the age of acquisition of words.")
    device = model.device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data = CHILDESDataset('./aoa_data/sent_words.json')
    dataloader = DataLoader(data, batch_size=batch_size)
    word_list_dict = open_word_list_csv()
    word_mapping = make_token_word_mappings(word_list_dict, tokenizer)
    logger.info(f"\n» Collecting model average surprisal values.")
    word_surprisals_n = get_batched_surprisal(model, tokenizer, model_type, dataloader, word_mapping, device)
    logger.info(f"\n» Fitting regression models using leave-one-out cross-validation.")
    mad_results = get_loo_mad_results(word_surprisals_n, word_list_dict)
    return word_surprisals_n, mad_results
