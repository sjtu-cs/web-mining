import numpy as np
import re
import itertools
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def numbers_to_vector(argument):
    list=[]
    for each in argument:
        switcher = {
         "0": [1,0,0,0,0],
         "1": [0,1,0,0,0],
         "2": [0,0,1,0,0],
         "3": [0,0,0,1,0],
         "4": [0,0,0,0,1],
               }
        list.append(switcher.get(each, [0, 0, 0, 0, 0]))
    return list


stop = {'ourselves', 'hers', 'between', 'yourself', 'again', 'there', 'about', 'once', 'during',
       'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of',
        'most', 'itself', 'other', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
        'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
        'were', 'her', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'to',
         'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them', 'and', 'been', 'have', 'in', 'will',
         'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'so', 'can', 'did', 'now', 'he', 'you',
         'herself', 'has', 'just', 'where', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
        'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here',
        'than', 'doe', 'character', 'rrb', 'lrb', 'story', 'one', 'ha', 'wa'}
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    #停用词
   # string = ' '.join([word for word in string.split() if word not in stop])
    #词形还原
    #lemmatizer = WordNetLemmatizer()
   # string = ''.join([lemmatizer.lemmatize(word) for word in string])
    return string.strip().lower()
def load_test_data_and_labels(lable_data_file):
    lable_examples = list(open(lable_data_file, "r", -1, "utf-8").readlines())
    lable_examples = [s.strip() for s in lable_examples]
    #temp2[0]))
    x_text = [clean_str(sent[2:]) for sent in lable_examples]
    lab = [sent[0] for sent in lable_examples]

    return [x_text, numbers_to_vector(lab)]

def load_data(lable_data_file):
    lable_examples = list(open(lable_data_file, "r", -1, "utf-8").readlines())
    lable_examples = [s.strip() for s in lable_examples]
    x_text = lable_examples
    x_text2 = [clean_str(sent) for sent in x_text]
    x_text=[sent for sent in x_text]
    return[x_text2,x_text]

def load_data_and_labels(d0_data_file, d1_data_file,d2_data_file,d3_data_file,d4_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    degree0_examples = list(open(d0_data_file, "r",-1,"utf-8").readlines())
    degree0_examples = [s.strip() for s in degree0_examples]
    degree1_examples = list(open(d1_data_file, "r",-1,"utf-8").readlines())
    degree1_examples = [s.strip() for s in degree1_examples]
    degree2_examples = list(open(d2_data_file, "r", -1, "utf-8").readlines())
    degree2_examples = [s.strip() for s in degree2_examples]
    degree3_examples = list(open(d3_data_file, "r", -1, "utf-8").readlines())
    degree3_examples = [s.strip() for s in degree3_examples]
    degree4_examples = list(open(d4_data_file, "r", -1, "utf-8").readlines())
    degree4_examples = [s.strip() for s in degree4_examples]
    # Split by words
    x_text = degree0_examples + degree1_examples+degree2_examples+degree3_examples+degree4_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    degree0_labels = [[1, 0, 0, 0, 0] for _ in degree0_examples]
    degree1_labels = [[0, 1, 0, 0, 0] for _ in degree1_examples]
    degree2_labels = [[0, 0, 1, 0, 0] for _ in degree2_examples]
    degree3_labels = [[0, 0, 0, 1, 0] for _ in degree3_examples]
    degree4_labels = [[0, 0, 0, 0, 1] for _ in degree4_examples]
    y = np.concatenate([degree0_labels, degree1_labels,degree2_labels,degree3_labels,degree4_labels], 0)
    return [x_text, y]

def load_data_and_labels2(data_file):
    datalist = list(open(data_file, "r", -1, "utf-8").readlines())
    datalist = [s.strip() for s in datalist]
    labels = [s[0] for s in datalist]
    records = [s[2:] for s in datalist]
    x_text = [clean_str(sent) for sent in records]
    lab_v = []
    for index in range(len(labels)):
        if labels[index] == '0':
            lab_v.append([1, 0, 0, 0, 0])
        elif labels[index] == '1':
            lab_v.append([0, 1, 0, 0, 0])
        elif labels[index] == '2':
            lab_v.append([0, 0, 1, 0, 0])
        elif labels[index] == '3':
            lab_v.append([0, 0, 0, 1, 0])
        elif labels[index] == '4':
            lab_v.append([0, 0, 0, 0, 1])
    return [x_text, lab_v]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]