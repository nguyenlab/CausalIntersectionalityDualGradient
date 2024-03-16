# Adapted from OSCAR Repo

import torch
from torch import nn

import logging
import math

from src.vilio.transformers.modeling_bert import (
    BertEmbeddings,
    BertSelfAttention,
    BertAttention,
    BertEncoder,
    BertLayer,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertLayerNorm,
    BertPreTrainedModel,
)

from src.vilio.transformers.activations import gelu, gelu_new, swish

### A) ACTIVATION FUNCS ###

logger = logging.getLogger(__name__)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_new": gelu_new,
    "mish": mish,
}


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


### B) BERT HELPERS ###


class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)
        self.output_attentions = config.output_attentions

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        history_state=None,
        attention_weights=None,
    ):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if attention_weights is None:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(
                query_layer,
                key_layer.transpose(-1, -2),
            )
            regularization_coef = math.sqrt(self.attention_head_size)
            attention_scores = attention_scores / regularization_coef
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
        else:
            attention_probs = attention_weights

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )

        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        input_tensor,
        attention_mask,
        head_mask=None,
        history_state=None,
        attention_weights=None,
    ):
        self_outputs = self.self(
            input_tensor,
            attention_mask,
            head_mask,
            history_state,
            attention_weights,
        )
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


### C) LAYER ###


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        history_state=None,
        attention_weights=None,
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            history_state,
            attention_weights,
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


### D) ENCODER ###


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [CaptionBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        encoder_history_states=None,
        attention_weights=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = (
                None if encoder_history_states is None else encoder_history_states[i]
            )
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                history_state,
                attention_weights,
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


### E) OSCAR MODEL ###


class BertO(BertPreTrainedModel):
    """Expand from BertModel to handle image region features as input
    "BertImgModel" > BertO
    """

    def __init__(self, config, img_feature_dim=2052):

        # Features + Positions (2048 + 4)
        config.img_feature_dim = img_feature_dim
        config.img_feature_type = "faster_r-cnn"
        config.code_voc = 512

        # Original Repo uses 0.3 dropout
        config.hidden_dropout_prob = 0.3

        super(BertO, self).__init__(config)

        # Added for integrated gradient
        config.output_attentions = True
        self.num_labels = 2
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info("BertImgModel Image Dimension: {}".format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, "use_img_layernorm"):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        if config.img_feature_type == "dis_code":
            self.code_embeddings = nn.Embedding(
                config.code_voc, config.code_dim, padding_idx=0
            )
            self.img_embedding = nn.Linear(
                config.code_dim, self.config.hidden_size, bias=True
            )
        elif config.img_feature_type == "dis_code_t":  # transpose
            self.code_embeddings = nn.Embedding(
                config.code_voc, config.code_dim, padding_idx=0
            )
            self.img_embedding = nn.Linear(
                config.code_size, self.config.hidden_size, bias=True
            )
        elif config.img_feature_type == "dis_code_scale":  # scaled
            self.input_embeddings = nn.Linear(
                config.code_dim, config.code_size, bias=True
            )
            self.code_embeddings = nn.Embedding(
                config.code_voc, config.code_dim, padding_idx=0
            )
            self.img_embedding = nn.Linear(
                config.code_dim, self.config.hidden_size, bias=True
            )
        else:
            self.img_embedding = nn.Linear(
                self.img_dim, self.config.hidden_size, bias=True
            )
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = BertLayerNorm(
                    config.hidden_size, eps=config.img_layer_norm_eps
                )

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        img_feats=None,
        encoder_history_states=None,
        # added
        attention_weights=None,
        is_logsoftmax=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        if encoder_history_states:
            assert (
                img_feats is None
            ), "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == "dis_code":
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == "dis_code_t":  # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == "dis_code_scale":  # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat(
                (
                    embedding_output,
                    img_embedding_output,
                ),
                1,
            )

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            encoder_history_states=encoder_history_states,
            # added
            attention_weights=attention_weights,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if is_logsoftmax:
            logits = self.logsoftmax(pooled_output)
        else:
            logits = pooled_output
        logits = self.softmax(logits)

        # calculate gradient if attention weight is given

        if attention_weights is not None:
            gradients = []
            num_samples = pooled_output.shape[0]
            for idx in range(num_samples):
                gradient = torch.autograd.grad(
                    torch.unbind(logits[idx, :]),
                    attention_weights,
                    retain_graph=True,
                )[0]
                gradients.append(gradient[idx])
            gradients = torch.stack(gradients)
            encoder_outputs = encoder_outputs + (gradients,)

        else:
            pass

        # add hidden_states and attentions if they are here
        outputs = (
            sequence_output,
            pooled_output,
            encoder_outputs[1:],
        )  # + encoder_outputs[1:]
        return outputs
