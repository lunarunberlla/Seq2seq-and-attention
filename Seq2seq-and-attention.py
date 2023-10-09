import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string

class DataGenerator:
    def __init__(self, vocabulary_size, seq_len):
        self.vocabulary_size = vocabulary_size
        self.seq_len = seq_len

    def randomString(self):
        """Generate a random string with the combination of uppercase letters """
        letters = string.ascii_uppercase
        return ''.join(random.choice(letters) for i in range(self.seq_len))

    def get_batch(self, batch_size):
        batched_examples = [self.randomString() for i in range(batch_size)]
        enc_x = [[ord(ch) - ord('A') + 1 for ch in list(exp)] for exp in batched_examples]
        y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
        dec_x = [[0] + e_idx[:-1] for e_idx in y]
        return (batched_examples, torch.tensor(enc_x), torch.tensor(dec_x), torch.tensor(y))



class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        out, (h0, c0) = self.encoder((x))
        return out, (h0, c0)

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x, hc):
        out, (h0, c0) = self.decoder((x), hc)
        return out, (h0, c0)

class Seq2seq(nn.Module):
    def __init__(self, hidden_size, vocabulary_size):
        super().__init__()
        self.embedding_enc = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
        self.embedding_dec = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)
        self.liner = nn.Linear(in_features=hidden_size, out_features=vocabulary_size)

    def forward(self, enc_x, dec_x):
        enc_x = self.embedding_enc(enc_x)
        dec_x = self.embedding_dec(dec_x)

        _, enc_hc = self.encoder(enc_x)
        out, dec_hc = self.decoder(dec_x, enc_hc)
        out = self.liner(out)
        normal_out = F.log_softmax(out, dim=-1)
        return normal_out

class Seq2SeqModel:
    def __init__(self, data_generator, hidden_size, batch_size):
        self.data_generator = data_generator
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.model = Seq2seq(hidden_size, data_generator.vocabulary_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimi = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, epochs):
        for i in range(epochs):
            _, enc_x, dec_x, y = self.data_generator.get_batch(self.batch_size)
            out = self.model(enc_x, dec_x)
            loss = 0
            for j in range(out.shape[0]):
                loss += self.loss_fn(out[j], y[j])
            loss /= out.shape[0]
            loss.backward()
            self.optimi.step()
            self.optimi.zero_grad()
            if (i + 1) % 200 == 0:
                print("epoch：{}, loss={}".format(i, loss))

    def predict(self, test_epoch):
        results = []
        for _ in range(test_epoch):
            batch_examples, enc_x, dec_x, y = self.data_generator.get_batch(1)
            enc_x = self.model.embedding_enc(enc_x)
            _, hc = self.model.encoder(enc_x)
            y = torch.zeros([1, 1], dtype=torch.int32)
            pred = ''
            for _ in range(self.data_generator.seq_len):
                with torch.no_grad():
                    y = self.model.embedding_dec(y)
                    out, hc = self.model.decoder(y, hc)
                    out = self.model.liner(out)
                    y = out.argmax(-1)
                pred += chr(ord('A') + y.item() - 1)
            results.append(int(pred == ''.join(i for i in reversed(batch_examples[0]))))
        return results


if __name__ == '__main__':
    vocabulary_size = len(string.ascii_uppercase) + 1
    hidden_size = 200
    seq_len = 5
    batch_size = 10

    data_generator = DataGenerator(vocabulary_size, seq_len)
    seq2seq_model = Seq2SeqModel(data_generator, hidden_size, batch_size)

    # 训练模型
    seq2seq_model.train(5000)

    # 预测
    results = seq2seq_model.predict(100)

    # 输出准确率
    print("acc={}%".format(sum(results)))

    #result:
    # epoch: 3999, loss: 0.06591387093067169
    # epoch: 4199, loss: 0.05470824986696243
    # epoch: 4399, loss: 0.0387391522526741
    # epoch: 4599, loss: 0.034858670085668564
    # epoch: 4799, loss: 0.030272463336586952
    # epoch: 4999, loss: 0.06778707355260849
    # acc = 99 %