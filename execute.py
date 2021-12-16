# -*- coding:utf-8 -*-
import os
import sys
import time
import torch
import seq2seqModel
import getConfig
import io
from tqdm import tqdm
import jieba
from torch import optim


jieba.setLogLevel(jieba.logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gConfig = {}
gConfig = getConfig.get_config(config_file='seq2seqChatbot.ini')
units = gConfig['layer_size']
BATCH_SIZE = gConfig['batch_size']
max_length_tar=20

EOS_token = 1
SOS_token = 0
MAX_LENGTH = 200


def preprocess_sentence(w):
    w = 'SOS ' + w + ' EOS'
    return w


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[preprocess_sentence(w)for w in l.split('\t')]
             for l in lines[:num_examples]]
    input_lang = Lang("ans")
    output_lang = Lang("ask")
    pairs = [list(reversed(p)) for p in pairs]
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs


def max_length(tensor):
    return max(len(t) for t in tensor)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def indexesFromSentence(lang, sentence):
    # print('-------------')
    # print(lang.word2index)
    # print(sentence)
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):

    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def read_data(path, num_examples):
    input_tensors = []
    target_tensors = []
    input_lang, target_lang, pairs = create_dataset(path, num_examples)
    for i in range(0, num_examples-1):
        try:
            input_tensor = tensorFromSentence(input_lang, pairs[i][0])
            target_tensor = tensorFromSentence(target_lang, pairs[i][1])
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
        except:
            break

    return input_tensors, input_lang, target_tensors, target_lang


input_tensor, input_lang, target_tensor, target_lang = read_data(
    gConfig['seq_data'], gConfig['max_train_data_size'])
hidden_size = 256

def train():
    # steps_per_epoch = len(input_tensor) // gConfig['batch_size']
    steps_per_epoch = 1
    checkpoint_dir = gConfig['model_data']
    checkpoint_prefix = os.path.join(checkpoint_dir, ".pt")
    start_time = time.time()
    encoder1 = seq2seqModel.Encoder(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = seq2seqModel.AttentionDencoder(
        hidden_size, target_lang.n_words, dropout_p=0.1).to(device)
    if os.path.exists(checkpoint_prefix):
        checkpoint = torch.load(checkpoint_prefix)
        encoder1.load_state_dict(checkpoint['modelA_state_dict'])
        attn_decoder1.load_state_dict(checkpoint['modelB_state_dict'])
        lg = checkpoint['epoch']
        print('加载模型成功!,epoch:{}'.format(lg))
    else:
        lg = 0
        print('没有模型,开始新建模型')
    max_data = gConfig['max_train_data_size']
    while True:
        start_time_epoch = time.time()
        total_loss = 0
        bar = tqdm(range(max_data-1),ascii=True,desc='数据处理')
        # range(0, max_data-1)
        for i in bar:
            inp = input_tensor[i]
            targ = target_tensor[i]
            batch_loss = seq2seqModel.train_step(inp, targ, encoder1, attn_decoder1, optim.SGD(
                encoder1.parameters(), lr=0.001), optim.SGD(attn_decoder1.parameters(), lr=0.001))
            total_loss += batch_loss
            # print('训练总步数:{} 最新每步loss {:.4f}'.format(i, batch_loss))
            bar.set_description('epoch:{}\tidx:{}\tloss:{:.4f}'.format(lg,i,batch_loss))
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps
        print(
            '训练总步数: {} 每步耗时: {} 最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                    step_loss))
        torch.save({'modelA_state_dict': encoder1.state_dict(),
                    'modelB_state_dict': attn_decoder1.state_dict(),
                    'epoch':lg
                    }, checkpoint_prefix)
        lg = lg + 1
        sys.stdout.flush()
        time.sleep(0.1)


# def predict(sentence)
def predict():
    print(input_lang.word2index)
    encoder = seq2seqModel.Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = seq2seqModel.AttentionDencoder(hidden_size,target_lang.n_words , dropout_p=0.1).to(device)
    checkpoint_dir = gConfig['model_data']
    checkpoint_prefix = os.path.join(checkpoint_dir, ".pt")
    checkpoint=torch.load(checkpoint_prefix)
    encoder.load_state_dict(checkpoint['modelA_state_dict'])
    decoder.load_state_dict(checkpoint['modelB_state_dict'])
    lg = checkpoint['epoch']
    print('这是训练了{}次的模型'.format(lg))


    while True:
        sentence = input('问: ')
        # 将语句使用结巴分词进行分词
        sentence = " ".join(jieba.cut(sentence))

        sentence = preprocess_sentence(sentence)
        input_tensor = tensorFromSentence(input_lang,sentence)
        input_length = input_tensor.size()[0]
        result = ''
        max_length=MAX_LENGTH
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        dec_input = torch.tensor([[SOS_token]], device=device)  # SOS
        dec_hidden = encoder_hidden
        decoder_attentions = torch.zeros(max_length, max_length)
        for t in range(max_length_tar):
            # encoder_outputs[t]
            predictions, dec_hidden, decoder_attentions = decoder(dec_input, dec_hidden, encoder_outputs)
            predicted_id,topi =predictions.data.topk(1)
            if topi.item() == EOS_token:
                result+='<EOS>'
                break
            else:
              result+=target_lang.index2word[topi.item()]+' '
            dec_input = topi.squeeze().detach()
        print(result)
        # return result

if __name__ == '__main__':
    predict()
    # train()
    # print(predict("呵呵"))
    # if len(sys.argv) - 1:
    #     gConfig = getConfig.get_config(sys.argv[1])
    # else:
    #     gConfig = getConfig.get_config()
    # print('\n>> Mode : %s\n' %(gConfig['mode']))
    # if gConfig['mode'] == 'train':
    #     train()
    # elif gConfig['mode'] == 'serve':
    #     print('Serve Usage : >> python3 app.py')
