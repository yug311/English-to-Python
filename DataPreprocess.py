
# Importing libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
import tokenize
import io
import re
from nltk.stem import PorterStemmer

import requests



# Retrieving the data
# !wget "https://drive.google.com/u/0/uc?id=1rHb0FQ5z5ZpaY2HpyFGY6CeyDG0kTLoO&export=download" -O english_python_data.txt

# Examining the dataset
# with open('english_python_data.txt',"r") as data_file:
#   print(data_file.readlines()[:5]) # Printing out the first 5 lines of the data

# Making a dataset
with open('english_python_data.txt',"r") as data_file:
  data_lines = data_file.readlines()
  dps = [] # List of dictionaries
  dp = None # The current problem and solution
  for line in data_lines:
    if line[0] == "#":
      if dp:
        dp['solution'] = ''.join(dp['solution'])
        dps.append(dp)
      dp = {"question": None, "solution": []}
      dp['question'] = line[1:]
    else:
      dp["solution"].append(line)

# converting the data to a table for easier viewing
dataset = pd.DataFrame(dps)

# Looking at the first question and the corresponding solution
# print(dataset.loc[0,'question'])
# print(dataset.loc[0,'solution'])

# Creating a class that holds the vocabulary mappings for each
# "Language" (English and Python in our case)

# Setting the SOS and EOS tokens
SOS_token = 1
EOS_token = 2

class Language:
  def __init__(self,name):
    self.name = name
    self.word2index = {'SOS':1,'EOS':2}
    self.word2count = {}
    self.index2word = {1:'SOS',2:'EOS'}
    self.n_words = 3

  # method to add sentences
  def addSentence(self, sentence):
    for word in sentence:
      self.addWord(word)

  # Method to add a word
  def addWord(self,word):
    if word not in self.word2index.keys():
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1

# Preprocessing the Python Code
# For this one, I just need to tokenize the Python code
indicies_to_ignore = [] # indicies that have broken Python code
tokenized_python = []
for i in range(dataset.shape[0]):
  try:
    solution = dataset.loc[i,'solution'].strip("\n ") # Stripping \n
    tokenized_code = [token.string for token in list(tokenize.generate_tokens(io.StringIO(solution).readline)) if token.string] # also removing empty characters
    tokenized_python.append(tokenized_code)
  except:
    indicies_to_ignore.append(i)
# print(f'Total Acceptable Examples: {len(tokenized_python)}')

# Dropping the indicies that were in the indicies_to_ignore list.
# Python code was written properly for me to utilize that example.
dataset_copy = dataset.copy()
dataset_copy.drop(indicies_to_ignore,inplace=True)
dataset_copy.reset_index(drop=True, inplace=True)

# Cleaning the English texts
cleaned_english = []
url_pattern = re.compile(r"https?://\S+")
tokenizer = get_tokenizer('basic_english')
stemmer = PorterStemmer()

for i in range(dataset_copy.shape[0]):
  sentence = dataset_copy.loc[i,'question']
  sentence = sentence.lower() # lowercasing everything
  sentence = sentence.strip("\n") # removing \n
  sentence = url_pattern.sub('',sentence) # Replacing any urls
  sentence = re.sub(r"([.!?])","",sentence) # removing any punctuation
  sentence = re.sub(r'^\d+ ',"",sentence)
  sentence = tokenizer(sentence) # tokenizing
  sentence = [stemmer.stem(word) for word in sentence] # Stemming
  cleaned_english.append(sentence)

# print(f'Number of Sentences: {len(cleaned_english)}')

# Zipping everything together to get a complete dataset
prepared_data = list(zip(cleaned_english,tokenized_python))

# Building the 2 vocabularies
english_lang = Language('english')
python_lang = Language('python')

# Populating the vocabularies
for (english, python) in prepared_data:
  english_lang.addSentence(english)
  python_lang.addSentence(python)

# Printing the number of words in each vocabulary
# print(f'Number of words in the English Vocabulary: {english_lang.n_words}')
# print(f'Number of words in the Python Vocabulary: {python_lang.n_words}')

english_lang.index2word[2]

# Getting the maximum length for both sequences
max_length_eng = 0
max_length_python = 0
avg_eng_length = 0
avg_python_length = 0
number_of_pairs = 0

for (english,python) in prepared_data:
  if len(english) > max_length_eng:
    max_length_eng = len(english)

  if len(python) > max_length_python:
    max_length_python = len(python)

  number_of_pairs += 1
  avg_eng_length += len(english)
  avg_python_length += len(python)

# Printing out the maximum lengths
# print(f'Maximum Length of an English Sentence: {max_length_eng}')
# print(f'Maximum Length of Python code: {max_length_python}')

# Printing out the average lengths
# print(f'Average Length of an English Sentence: {avg_eng_length / number_of_pairs}')
# print(f'Average Length of Python code: {avg_python_length / number_of_pairs}')


# Creating the dataset
final_prepared_data = []
for (english,python) in prepared_data:
  english_sentence = ['SOS']
  english_sentence.extend(english)
  english_sentence.append('EOS')

  python_sentence = ['SOS']
  python_sentence.extend(python)
  python_sentence.append('EOS')

  # Tokenizing the english sentences
  tokenized_english = []
  for word in english_sentence:
    tokenized_english.append(english_lang.word2index[word])

  # Tokenizing the python sentences
  tokenized_python = []
  for word in python_sentence:
    tokenized_python.append(python_lang.word2index[word])

  # Getting the padding needed
  """

  You might want to play around with the max length!


  """
  padding_needed_eng = 24 - len(tokenized_english)
  padding_needed_python = 64 - len(tokenized_python)

  # Checking if I need to trim
  if padding_needed_eng <= 0:
    tokenized_english = tokenized_english[:23]
    tokenized_english.append(EOS_token)
    tokenized_english = torch.from_numpy(np.array(tokenized_english))
  else:
    tokenized_english = nn.functional.pad(torch.from_numpy(np.array(tokenized_english)),(0,padding_needed_eng))

  # Checking if I need to trim for Python
  if padding_needed_python <= 0:
    tokenized_python = tokenized_python[:63]
    tokenized_python.append(EOS_token)
    tokenized_python = torch.from_numpy(np.array(tokenized_python))
  else:
    tokenized_python = nn.functional.pad(torch.from_numpy(np.array(tokenized_python)),(0,padding_needed_python))

  final_prepared_data.append((tokenized_english.numpy(),tokenized_python.numpy()))


