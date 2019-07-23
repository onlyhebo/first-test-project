import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from hmt.modules import LayerNorm


class Mymodel(nn.Module):
    def __init__(self, dictionary, opt, tgt_size):
        super().__init__()
        self.model = nn.LSTM(opt.src_word_vec_size, dropout=opt.lstm_dropout,
                             hidden_size=opt.hidden_size, bidirectional=True)
        self.opt = opt
        self.label = nn.Linear(opt.hidden_size*2, tgt_size, bias=True)
        self.set_model_parameters()
        self.embed_tokens = self.build_embedding(dictionary)
        self.layer_norm = LayerNorm(opt.src_word_vec_size)
        self.embed_dropout = nn.Dropout(opt.embed_dropout)
        self.final_dropout = nn.Dropout(opt.final_dropout)
        self.bn = nn.BatchNorm1d(opt.hidden_size*2)

    def build_embedding(self, dictionary):
        num_embedding = len(dictionary)
        padding_idx = dictionary.pad()
        emb = Embedding(num_embedding, self.opt.src_word_vec_size, padding_idx)
        return emb

    def forward(self, x):
        src_tokens = x['src_tokens']  # batch_size * sequence_length
        src_tokens = self.embed_tokens(src_tokens)  # batch_size * sequence_length * embed_dim
        self.embed_dropout(src_tokens)
        src_tokens = self.layer_norm(src_tokens)
        src_tokens.transpose_(1, 0)  # sequences_length, batch_size, embed_dim
        packed = pack_padded_sequence(src_tokens, x['src_lengths'])
        output, (h_n, c_n) = self.model(packed)
        # output, output_lengths = pad_packed_sequence(output)  # output [seq_len, batch, num_directions*hidden_size]
        # h_n/c_n [num_layers*num_directions, batch, hidden_size]
        last_tensor = h_n.transpose(0, 1).contiguous().view(src_tokens.size(1), -1)
        label_input = self.bn(last_tensor)  # Accelerate the training of deep neural nets
        label_input = self.final_dropout(label_input)
        final_output = self.label(label_input)
        return final_output

    def set_model_parameters(self):
        for p in self.parameters():
            p.data.uniform_(-0.1, 0.1)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

