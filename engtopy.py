import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import gc
from Models import Encoder, Decoder
from DataPreprocess import final_prepared_data, python_lang
import time

# Setting the device for model
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device("mps")

# Splitting the data into training, testing, and validation
training, testing = train_test_split(final_prepared_data,test_size=0.2,random_state=42,shuffle=True)

# Splitting the data into inputs and outputs
inputs_train = torch.Tensor([pair[0] for pair in training])
outputs_train = torch.Tensor([pair[1] for pair in training])
inputs_test = torch.Tensor([pair[0] for pair in testing])
outputs_test = torch.Tensor([pair[1] for pair in testing])

# Creating the Tensor Datasets
training_dataset = TensorDataset(inputs_train,outputs_train)
testing_dataset = TensorDataset(inputs_test,outputs_test)

# Building the data loaders
training_dataloader = DataLoader(training_dataset,batch_size=32,shuffle=True)
testing_dataloader = DataLoader(testing_dataset,batch_size=32,shuffle=True)

NEG_INFTY = -1e9
max_sequence_length_eng = 24
max_sequence_length_py = 64

def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length_py, max_sequence_length_py] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length_eng, max_sequence_length_eng] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length_py, max_sequence_length_py] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length_py, max_sequence_length_eng] , False)

    for idx in range(num_sentences):
        eng_sentence_length, kn_sentence_length = torch.nonzero(eng_batch[idx]).size(0), torch.nonzero(kn_batch[idx]).size(0)
        eng_chars_to_padding_mask = np.arange(eng_sentence_length, max_sequence_length_eng)
        kn_chars_to_padding_mask = np.arange(kn_sentence_length, max_sequence_length_py)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

def translate(english_sentence, y=None):
  encoder.eval()
  decoder.eval()
  python_sentence = ""
  tokenized_python = torch.zeros(1, 64)
  tokenized_python[0][0] = 1
  # tokenized_python = y

  for i in range(64):

    # assume english sentence is tokenized already
    encoder_mask, decod_mask_self, decod_mask_cross = create_masks(english_sentence, tokenized_python)
    outputs = encoder(english_sentence.int(), encoder_mask)
    predicts = decoder(outputs, tokenized_python.int(), decod_mask_self, decod_mask_cross)

    next_token_probs = predicts[0][i]
    index = torch.argmax(next_token_probs).item()
    if index == 0:
      print("break")
      continue

    token = python_lang.index2word[index]
    python_sentence += token + " "
    print(tokenized_python)
    print(index)

    if i < 50:
      if y is not None:
        tokenized_python[0][i + 1] = y[0][i+1]
      else:
        tokenized_python[0][i + 1] = index
    else:
      break

  return python_sentence

encoder = Encoder()
decoder = Decoder()
encoder.train()
decoder.train()
criterion = nn.CrossEntropyLoss(reduction='sum')
encoder_optim = optim.AdamW(encoder.parameters(), lr=0.001)
decoder_optim = optim.AdamW(decoder.parameters(), lr=0.001)
schedulerm = torch.optim.lr_scheduler.StepLR(encoder_optim, step_size=10, gamma=0.1)
schedulerd = torch.optim.lr_scheduler.StepLR(decoder_optim, step_size=10, gamma=0.1)
encoder.to(device)
decoder.to(device)
encoder = torch.compile(encoder)
decoder = torch.compile(decoder)

torch.set_float32_matmul_precision('high')

checkpoint_path = 'checkpoint.pth'

try:
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder_optim.load_state_dict(checkpoint['encoder_optim'])
    decoder_optim.load_state_dict(checkpoint['decoder_optim'])
    schedulerm.load_state_dict(checkpoint['schedulerm'])
    schedulerd.load_state_dict(checkpoint['schedulerd'])
except FileNotFoundError:
    pass

print("Starting training")
for epoch in range(3):
  total_loss = 0
  for i, (x, y) in enumerate(training_dataloader):  # Remember to call x.int()
    d0 = time.time()
    x = x.to(device)
    y = y.to(device)
    batch_size = len(y)

    rolled_y = torch.roll(y, shifts=-1, dims=1)
    rolled_y[:, -1] = 0
    y_reshaped = rolled_y.reshape(batch_size * 64)

    # Forward
    encoder_mask, decod_mask_self, decod_mask_cross = create_masks(x, y)
    encoder_outputs = encoder(x.int(), encoder_mask)

    outputs = decoder(encoder_outputs, y.int(), decod_mask_self, decod_mask_cross)
    outputs_reshaped = outputs.reshape(batch_size * 64, 6873)  # num python tokens

    loss = criterion(outputs_reshaped, y_reshaped.long())
    total_loss += loss.item()

    # Backward
    loss.backward()
    norm1 = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
    norm2 = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    encoder_optim.step()
    decoder_optim.step()
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    d1 = time.time()


    print(f'loss:{loss.item()} | time:{(d1-d0)*1000:.2f}')



    if(i % 10 == 0 and i > 0):
      print(f'epoch {epoch}, step: {i}')

    # Explicitly delete tensors to free up memory
    del encoder_outputs, outputs, outputs_reshaped, y_reshaped, encoder_mask, decod_mask_self, decod_mask_cross
    torch.cuda.empty_cache()  # If using GPU
    gc.collect()

  print(f'epoch {epoch}: loss= {total_loss / len(training_dataloader)}')

# Save the model
torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'encoder_optim': encoder_optim.state_dict(),
    'decoder_optim': decoder_optim.state_dict(),
    'schedulerm': schedulerm.state_dict(),
    'schedulerd': schedulerd.state_dict(),
}, checkpoint_path)


x, y = training_dataset[0]
x = x.unsqueeze(0)
y = y.unsqueeze(0)
prev_y = y
translate(x)
