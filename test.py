import nltk
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from train import model,vocab
model.load_state_dict(torch.load("model_weight.pth"))



