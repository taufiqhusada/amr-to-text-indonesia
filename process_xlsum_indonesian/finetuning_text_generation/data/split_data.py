import math

# amr
with open('full.amr.txt', encoding='utf8') as f:
    data = f.readlines()

len_total = len(data)
len_train_split = math.floor(0.9*len_total)
len_dev_split = math.floor(0.05*len_total)
len_test_split = len_total - len_train_split - len_dev_split

train = data[0:len_train_split]
dev = data[len_train_split:len_dev_split + len_train_split + 1]
test = data[len_dev_split + len_train_split + 1:]

print(len(train), len(dev), len(test))

with open('train.amr.txt', encoding='utf8', mode='w+') as f:
    for item in train:
        f.write(item)
with open('dev.amr.txt', encoding='utf8', mode='w+') as f:
    for item in dev:
        f.write(item)
with open('test.amr.txt', encoding='utf8', mode='w+') as f:
    for item in test:
        f.write(item)

# sentence
with open('full.sent.txt', encoding='utf8') as f:
    data = f.readlines()

len_total = len(data)
len_train_split = math.floor(0.9*len_total)
len_dev_split = math.floor(0.05*len_total)
len_test_split = len_total - len_train_split - len_dev_split

train = data[0:len_train_split]
dev = data[len_train_split:len_dev_split + len_train_split + 1]
test = data[len_dev_split + len_train_split + 1:]

print(len(train), len(dev), len(test))

with open('train.sent.txt', encoding='utf8', mode='w+') as f:
    for item in train:
        f.write(item)
with open('dev.sent.txt', encoding='utf8', mode='w+') as f:
    for item in dev:
        f.write(item)
with open('test.sent.txt', encoding='utf8', mode='w+') as f:
    for item in test:
        f.write(item)


