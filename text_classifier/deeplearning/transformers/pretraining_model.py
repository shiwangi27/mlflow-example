import torch
import torch.nn as nn

from transformers import BertModel

class Transformer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout,
                 causal=False):
        super(Transformer, self).__init__()

        self.causal = causal
        self.token_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms1, self.layer_norms2 = nn.ModuleList(), nn.ModuleList()

        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embedding_dim)))

            self.layer_norms1.append(nn.LayerNorm(embedding_dim, eps=1e-12))
            self.layer_norms2.append(nn.LayerNorm(embedding_dim, eps=1e-12))

    def forward(self, x, padding_mask=None):
        """ Input has shape [seq length, batch] """
        print("x", x)
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.token_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms1, self.attentions,
                                                                       self.layer_norms2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config.embedding_dim, config.hidden_dim, config.num_embeddings,
                                       config.num_max_positions, config.num_heads, config.num_layers,
                                       config.dropout, causal=not config.mlm)
        self.lm_head = nn.Linear(config.embedding_dim, config.num_embeddings, bias=False)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.token_embeddings.weight

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, labels=None, padding_mask=None):
        hidden_states = self.transformer(x, padding_mask)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[:-1] if self.transformer.causal else logits
            shift_labels = labels[1:] if self.transformer.causal else labels
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss

        return logits
