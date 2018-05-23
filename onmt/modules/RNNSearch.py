import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from onmt.Models import RNNDecoderState

class RNNSearchGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(RNNSearchGRU, self).__init__()

        self.input_weights = nn.Linear(embed_dim, hidden_dim*2)
        self.hidden_weights = nn.Linear(hidden_dim, hidden_dim*2)
        self.ctx_weights = nn.Linear(hidden_dim*2, hidden_dim*2)

        self.input_in = nn.Linear(embed_dim, hidden_dim)
        self.hidden_in = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_in = nn.Linear(hidden_dim*2, hidden_dim)


    def forward(self, trg_word, prev_s, ctx):
        '''
        trg_word : B x E
        prev_s   : B x H
        ctx      : B x 2*H
        '''
        gates = self.input_weights(trg_word) + self.hidden_weights(prev_s) + self.ctx_weights(ctx) # B, 2H
        reset_gate, update_gate = gates.chunk(2,1)

        reset_gate = F.sigmoid(reset_gate) # B, H
        update_gate = F.sigmoid(update_gate) # B, H

        prev_s_tilde = self.input_in(trg_word) + self.hidden_in(prev_s) + self.ctx_in(ctx)
        prev_s_tilde = F.tanh(prev_s_tilde) # B, H

        prev_s = torch.mul((1-reset_gate), prev_s) + torch.mul(reset_gate, prev_s_tilde) # B, H
        return prev_s

class RNNSearchAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(RNNSearchAttention, self).__init__()

        self.enc_h_in = nn.Linear(hidden_dim*2, hidden_dim)
        self.prev_s_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, enc_h, prev_s):
        '''
        enc_h  : S, B, 2*H
        prev_s : B, H
        '''
        seq_len = enc_h.size(1)

        enc_h_in = self.enc_h_in(enc_h) # S, B, H
        prev_s = self.prev_s_in(prev_s).unsqueeze(1)  # B, 1, H

        h = F.tanh(enc_h_in + prev_s.expand_as(enc_h_in)) # S, B, H
        h = self.linear(h)  # S, B, 1

        alpha = F.softmax(h)
        ctx = torch.bmm(alpha.transpose(0,1).transpose(2,1), enc_h.transpose(0,1)).squeeze(1) # B, 2H

        return ctx, alpha

class RNNSearchDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout, embeddings):
        super(RNNSearchDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = embeddings
        self.attention = RNNSearchAttention(hidden_dim)
        self.decoder_cell = RNNSearchGRU(embed_dim, hidden_dim)
        self.maxout_hidden_in = nn.Linear(hidden_dim, 2*hidden_dim)
        self.maxout_prev_y_in = nn.Linear(embed_dim, 2*hidden_dim)
        self.maxout_c_in = nn.Linear(2*hidden_dim, 2*hidden_dim)

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        # CHECKS
        assert isinstance(state, RNNDecoderState)

        target_len, batch_size = tgt.size(0), tgt.size(1)
        dec_h = Variable(torch.zeros(target_len, batch_size, self.hidden_dim))
        if torch.cuda.is_available():
            dec_h = dec_h.cuda()

        target = self.embed(tgt) # S, B, E
        hidden = state.hidden[0]
        attns = {"std": []}
        for i in range(target_len):
            ctx, alpha = self.attention(memory_bank, hidden)
            attns["std"] += [alpha.squeeze(2)]
            hidden = self.decoder_cell(target[i], hidden, ctx)
            maxout_t = self.maxout_c_in(ctx) + self.maxout_hidden_in(hidden) + self.maxout_prev_y_in(target[i])
            maxout_t = maxout_t.view(maxout_t.size(0), maxout_t.size(1), maxout_t.size(2)/2, 2)
            maxout_t = maxout_t.max(-1)[0] # B, H
            dec_h[i] = maxout_t
        state.update_state(hidden)
        attns["std"] = torch.stack(attns["std"]).transpose(1,2) # tgt_len, batch, src_len
        return dec_h, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        return RNNSearchDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))

class RNNSearchDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate

    @property
    def _all(self):
        return self.hidden

    def update_state(self, rnnstate):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])