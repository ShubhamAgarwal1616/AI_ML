import numpy as np
from data import train_data, test_data
from rnn import RNN
import random

# Create the vocabulary.
vocab = list(set([w.strip() for text in train_data.keys() for w in text.split(' ') if w.strip() != '']))
vocab_size = len(vocab)
# print('%d unique words found' % vocab_size) # 256 unique words found


# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
# print(word_to_idx['good']) # 16 (this may change)
# print(idx_to_word[0]) # sad (this may change)

languages = sorted(list(set(train_data.values())))
lang2idx = {lang: i for i, lang in enumerate(languages)}  # {'english':0, 'french':1, 'german':2}
output_size = len(languages)

def createInputs(text):
  '''
  Returns an array of one-hot vectors representing the words
  in the input text string.
  - text is a string
  - Each one-hot vector has shape (vocab_size, 1)
  '''
  inputs = []
  for w in text.split(' '):
    w = w.strip()
    if w == "" or w not in word_to_idx:   # 🔹 skip blanks or unknown words
      continue
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs

# print(createInputs("good vas"))

# Initialize our RNN!
rnn = RNN(vocab_size, output_size)

inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)
probs = RNN.softmax(out)
print(probs) # [[0.50000095], [0.49999905]]


def label_to_onehot(label):
    num_classes = len(languages)
    onehot = np.zeros((num_classes, 1))
    onehot[lang2idx[label]] = 1
    return onehot

def processData(data, backprop=True):
    '''
    Returns the RNN's loss and accuracy for the given data.
    - data is a dictionary mapping text -> language string.
    - backprop determines if the backward phase should be run.
    '''
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, lang in items:
        inputs = createInputs(x)
        target = lang2idx[lang]   # 🔹 convert "english" -> 0, "german" -> 1, etc.

        # Forward
        out, _ = rnn.forward(inputs)
        probs = RNN.softmax(out)

        # Loss / accuracy
        target_onehot = label_to_onehot(lang)
        loss += RNN.cross_entropy_loss(probs, target_onehot)
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Gradient of loss wrt y
            d_L_d_y = probs.copy()
            d_L_d_y[target] -= 1

            # Backward
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


for epoch in range(200):
  train_loss, train_acc = processData(train_data)

  if epoch % 20 == 19:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

    test_loss, test_acc = processData(test_data, backprop=False)
    print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))


print("\n\nFinal Result:")

idx2lang = {i: lang for lang, i in lang2idx.items()}
items = list(test_data.items())
random.shuffle(items)
print(lang2idx)
for x, lang in items:
  inputs = createInputs(x)
  out, _ = rnn.forward(inputs)
  probs = RNN.softmax(out)

  pred_idx = int(np.argmax(probs))         # index of max probability
  pred_lang = idx2lang[pred_idx]           # map index -> language name

  print(f"{x} - actual: {lang}, predicted: {pred_lang}")
  print("probs:", probs.T, "\n")     