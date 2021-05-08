from data_processing import *
from TextClassificationModel import *
import pandas
import gensim
from gensim import models
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to("cpu"), text_list.to("cpu"), offsets.to("cpu")


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()
            print("loss.item", loss.item())

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()


if __name__ == '__main__':
    imdb_df = pandas.read_csv("IMDB Dataset.csv")
    #print(imdb_df.head())
    #process(imdb_df)
    imdb_df["sentiment"] = imdb_df["sentiment"].replace({"negative": 0, "positive": 1})
    train_data = imdb_df[:40000]
    test_data = imdb_df[10000:].reset_index(drop=True)

    train_iter = list(train_data.itertuples(index=False, name=None))
    test_iter = list(test_data.itertuples(index=False, name=None))
    tokenizer = get_tokenizer("basic_english")
    counter = Counter()
    for (line, label) in train_iter:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1)

    text_pipeline = lambda x: [0 if token not in embeddings.index2word else embeddings.index2word.index(token) for token in tokenizer(x)]
    label_pipeline = lambda x: int(x)

    dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

    embedding_file = "glove.6B.100d.txt"
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    weights = torch.FloatTensor(embeddings.vectors)

    num_class = len(set([label for (text, label) in train_iter]))
    vocab_size = len(vocab)
    emsize = 100
    model = TextClassificationModel(vocab_size, emsize, num_class, weights).to("cpu")

    epochs = 10
    LR = 0.01
    batch_size = 64
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_dataset = list(train_iter)
    test_dataset = list(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)

        labels = {0: "negative", 1: "positive"}
        text_str_pos = "Confidently directed, dark, brooding, and packed with impressive action sequences and a complex story, The Dark Knight includes a career-defining turn from Heath Ledger as well as other Oscar worthy performances, TDK remains not only the best Batman movie, but comic book movie ever created."
        text_str_pos2 = "It is the best movie I can remember I've watched while I was a kid!"
        text_str_pos3 = "This movie, and trilogy in general, is a cinematic (and literary) masterpiece, and simply refuses to get old. I love this movie so much it's become a tradition to watch the Lord of the Rings series at least once a year! It's just as good as Christmas"
        text_str_pos4 = "The greatest tragedy of the human race is that they will never make a better movie than Return of the King."
        text_str_pos5 = "I thought this was a very good movie. Very well filmed and acted with amazing sets and scenery. My only complaint was that it went on too long. It should have finished at Mount Doom the rest of the scenes were not necessary."
        text_str_neg = "Before walking into this movie , i really had great expectations, what with the rave reviews the movie was being lavished with and the seemingly awesome trailer. And then came the moment of truth,the movie was very ordinary. I saw nothing special in this movie for me to even remotely praise it. It was long, many gaps in the plot in addition to scenes and story lines which seemed senseless and had no follow -up. Plus, there really was nothing artistic about it and it didn't actually have the enjoyable cinematic experience moviegoers would want to relish in. This was further aggravated with the poor lines in the film. In short the movie didn't get me all excited or sucked me into it. Throughout the entire film I was hardly able to connect to the film. That said however I wouldn't recommend folks to not watch it but just an advice to keep your hopes to a minimal point."
        text_str_neg2 = "Extremely boring and long movie with poor storyline. What is wrong with ending? It felt like 4 endings put together one after one, I swear he made 4 different endings and put them all in this movie. The action and special effects were great tho."
        text_str_neg3 = "I am very disappointed because up until now IMDb has never recommended so much a movie I consider unsatisfying. How can you put this movie as #1 ? Do you really think this is the best movie - if it is I should never watch cinema again !"
        text_str_neg4 = "Although suffering from a muddled plot, ill-defined characters, a typically uncharismatic performance by Christian Bale, and a lack of energy, The Dark Knight proved to be on of 2008's greatest commercial and critical successes."
        text_str_neg5 = "Sorry, people, despite all the unbelievable hoopla, this movie is terrible."
        #print("This is a %s sentence." %labels[predict(text_str_pos, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_pos2, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_pos3, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_pos4, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_pos5, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_neg, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_neg2, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_neg3, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_neg4, text_pipeline)])
        #print("This is a %s sentence." %labels[predict(text_str_neg5, text_pipeline)])

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader)
    print("test accuracy {:8.3f}".format(accu_test))
