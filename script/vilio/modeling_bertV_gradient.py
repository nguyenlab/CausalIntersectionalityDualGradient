# Code adapted from MMF, LXMERT, VisualBERT

from copy import deepcopy

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss, Softmax
from torch.nn.functional import relu

from src.vilio.file_utils import cached_path
from src.vilio.transformers.activations import gelu, gelu_new, swish
from src.vilio.transformers.modeling_bert_gradient import (
    BertConfig,
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BertOutput,
    BertIntermediate,
    BertSelfOutput,
    BertEncoder,
    BertPreTrainedModel,
    BertForPreTraining,
)


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


### B) VISUALCONFIG (only used in pretraining) ###

CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"
TF_WEIGHTS_NAME = "model.ckpt"


class VisualConfig(object):
    VISUAL_LOSSES = ["feat"]

    def __init__(self, l_layers=12, x_layers=5, r_layers=0):
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers

        self.visual_feat_dim = 2048
        self.visual_pos_dim = 4

        self.obj_id_num = 1600
        self.attr_id_num = 400

        self.visual_losses = self.VISUAL_LOSSES
        self.visual_loss_config = {
            "obj": (self.obj_id_num, "ce", (-1,), 1 / 0.15),
            "attr": (self.attr_id_num, "ce", (-1,), 1 / 0.15),
            "feat": (2048, "l2", (-1, 2048), 1 / 0.15),
        }

    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim


VISUAL_CONFIG = VisualConfig()

### C) EMBEDDINGS ###
# a) BertEmbeddings as imported
# c) VisioLing. embeddings

# Same as batch norm, but statistics for the whole layer, not just batch
BertLayerNorm = torch.nn.LayerNorm


class BertVisioLinguisticEmbeddings(BertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.token_type_embeddings_visual = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings_visual = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def initialize_visual_from_pretrained(self):
        self.token_type_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.token_type_embeddings.weight.data), requires_grad=True
        )
        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        visual_embeddings=None,
        visual_embeddings_type=None,
        position_embeddings_visual=None,
        image_text_alignment=None,
    ):
        """
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        If we define visual_embeddings, we must also define visual_embeddings_type, which is similar to token_type_ids in BERT.
        """

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if visual_embeddings is not None:
            visual_embeddings = self.projection(visual_embeddings)

            token_type_embeddings_visual = self.token_type_embeddings_visual(
                visual_embeddings_type
            )

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the
                # image feature. -1 is the padding value.
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # position_embeddings_visual
                # = Batch x image_length x alignment length x dim
                position_embeddings_visual = self.position_embeddings(
                    image_text_alignment
                ) * image_text_alignment_mask.to(
                    dtype=next(self.parameters()).dtype
                ).unsqueeze(
                    -1
                )
                position_embeddings_visual = position_embeddings_visual.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(
                    dtype=next(self.parameters()).dtype
                ).sum(2)
                image_text_alignment_mask[
                    image_text_alignment_mask == 0
                ] = 1  # Avoid devide by zero error
                position_embeddings_visual = (
                    position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1)
                )

                position_ids_visual = torch.zeros(
                    *visual_embeddings.size()[:-1], dtype=torch.long
                ).cuda()

                # When fine-tuning the detector , the image_text_alignment is
                # sometimes padded too long.
                if position_embeddings_visual.size(1) != visual_embeddings.size(1):
                    assert position_embeddings_visual.size(1) >= visual_embeddings.size(
                        1
                    )
                    position_embeddings_visual = position_embeddings_visual[
                        :, : visual_embeddings.size(1), :
                    ]

                position_embeddings_visual = (
                    position_embeddings_visual
                    + self.position_embeddings_visual(position_ids_visual)
                )
            else:

                position_ids_visual = torch.zeros(
                    *visual_embeddings.size()[:-1], dtype=torch.long
                ).cuda()

                position_embeddings_visual = self.position_embeddings_visual(
                    position_ids_visual
                )

            v_embeddings = (
                visual_embeddings
                + position_embeddings_visual
                + token_type_embeddings_visual
            )

            # Concate the two:
            embeddings = torch.cat(
                (embeddings, v_embeddings), dim=1
            )  # concat the visual embeddings after the attentions

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


### D) Final VB Model ###


class BertV(BertPreTrainedModel):
    """
    This class is a merger of MMFs VisualBERTBase, VisualBERT, VisualBERTForClassification & the original VisualBERT repo.
    """

    def __init__(
        self,
        config,
        visual_embedding_dim=2048,
        embedding_strategy="plain",
        bypass_transformer=False,
        output_attentions=True,
        output_hidden_states=True,
        layeravg=False,
    ):
        # Manual config changes:
        config.hidden_act = "gelu"

        config.bypass_transformer = bypass_transformer
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        config.visual_embedding_dim = visual_embedding_dim
        config.embedding_strategy = embedding_strategy
        config.layeravg = layeravg

        self.config = config

        super().__init__(config)
        print(config)

        self.layeravg = config.layeravg
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.bypass_transformer = config.bypass_transformer

        self.embeddings = BertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        if self.bypass_transformer:
            self.additional_layer = BertLayer(config)

        self.fixed_head_masks = [None for _ in range(len(self.encoder.layer))]

        # Taking hidden states from all hidden layers
        if self.layeravg:
            self.dropout = nn.Dropout(p=0.2)
            n_weights = config.num_hidden_layers + 1
            weights_init = torch.zeros(n_weights).float()
            weights_init.data[:-1] = -3
            self.layer_weights = torch.nn.Parameter(weights_init)

        self.softmax = Softmax(dim=1)

        self.init_weights()

    def extract_gradients(
        self, pooled_output: torch.Tensor, attention_weights: torch.Tensor
    ):
        logits = self.softmax(pooled_output)
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
        return gradients

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        num_features=100,
        attention_weights=None,
    ):

        # Manually create visual_embeddings_type
        if visual_embeddings_type is None:
            visual_embeddings_type = torch.ones(
                (visual_embeddings.shape[0], visual_embeddings.shape[1]),
                dtype=torch.long,
            ).cuda()

        # Note: We add 100 here, as it is the amount of features for HM
        # attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1] + num_features)).cuda()
        image_mask = torch.ones(
            (input_ids.shape[0], num_features), dtype=torch.long
        ).cuda()

        attention_mask = torch.cat((attention_mask, image_mask), dim=-1).cuda()

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            position_embeddings_visual=position_embeddings_visual,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
        )

        if self.bypass_transformer and visual_embeddings is not None:
            assert (
                not self.output_hidden_states
            )  # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[
                :, :, :text_length, :text_length
            ]

            encoded_layers = self.encoder(
                text_embedding_output,
                text_extended_attention_mask,
                self.fixed_head_masks,
                attention_weights=attention_weights,
            )

            atts = encoded_layers[2]
            intermediates = (atts,)

            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(
                new_input, extended_attention_mask
            )
            pooled_output = self.pooler(final_sequence_output)

            if attention_weights is not None:
                gradients = self.extract_gradients(
                    pooled_output,
                    attention_weights,
                )
                intermediates = (atts, gradients)
            else:
                pass

            return final_sequence_output, pooled_output, intermediates

        else:

            encoded_layers = self.encoder(
                embedding_output,
                extended_attention_mask,
                self.fixed_head_masks,
                output_hidden_states=self.output_hidden_states,
                attention_weights=attention_weights,
            )

            atts = encoded_layers[2]
            intermediates = (atts,)

            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)

            if self.layeravg:
                hidden_layers = encoded_layers[1]

                cls_outputs = torch.stack(
                    [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
                )

                sequence_output = (
                    torch.softmax(self.layer_weights, dim=0) * cls_outputs
                ).sum(-1)

            if attention_weights is not None:
                gradients = self.extract_gradients(
                    pooled_output,
                    attention_weights,
                )
                intermediates = (atts, gradients)
            else:
                pass

            return sequence_output, pooled_output, intermediates


### E) PRETRAINING ###


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertVisualObjHead(nn.Module):
    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Decide the use of visual losses
        visual_losses = visual_losses.split(",")
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {
                key: nn.Linear(
                    config.hidden_size, VISUAL_CONFIG.visual_loss_config[key][0]
                )
                for key in self.visual_losses
            }
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertVPretraining(nn.Module):
    """
    Mix of LXMERT Pretraining, MMF Pretraining, VisualBERT Pretraining.
    """

    def __init__(
        self,
        tr_name="bert-base-uncased",
        visual_losses="",
        task_obj_predict=False,
        task_matched=False,
    ):
        super().__init__()

        self.task_obj_predict = task_obj_predict
        self.task_matched = task_matched

        self.tr_name = tr_name

        self.bert = BertV.from_pretrained(
            self.tr_name, output_hidden_states=True, layeravg=False
        )

        self.vocab_size = self.bert.config.vocab_size

        bert_masked_lm = BertForPreTraining.from_pretrained(self.tr_name)

        self.cls = deepcopy(bert_masked_lm.cls)

        # LXMERTs Visn
        if self.task_obj_predict:
            self.obj_predict_head = BertVisualObjHead(self.bert.config, visual_losses)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def init_weights(self):
        if self.tr_name is None:
            # No pretrained model, init weights
            self.bert.init_weights()
            self.cls.apply(self.bert.init_weights)

        self.tie_weights()

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them
        instead.
        """
        self.bert._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
        matched_label=None,
        obj_labels=None,
        num_features=100,
    ):
        sequence_output, pooled_output = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            position_embeddings_visual,
            visual_embeddings_type,
            image_text_alignment,
            num_features,
        )

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        total_loss = 0
        losses = ()

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores.contiguous().view(-1, self.vocab_size),
                masked_lm_labels.contiguous().view(-1),
            )

            total_loss += masked_lm_loss
            losses += (masked_lm_loss.detach(),)

        if matched_label is not None and self.task_matched:
            matched_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2), matched_label.view(-1)
            )
            total_loss += matched_loss
            losses += (matched_loss.detach(),)

        # Implement LXMERTS Visn Losses
        if obj_labels is not None and self.task_obj_predict:
            loss_fcts = {
                "l2": SmoothL1Loss(reduction="none"),
                "ce": CrossEntropyLoss(ignore_index=-1, reduction="none"),
            }
            total_visn_loss = 0.0

            # Take visn output
            visn_output = sequence_output.clone()
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in VISUAL_CONFIG.visual_losses:
                label, mask_conf = obj_labels[key]

                # Add 0's to label at beginning for lang output
                lang_label = torch.zeros(
                    (label.size(0), 128, label.size(2)), dtype=torch.long
                ).cuda()
                label = torch.cat((lang_label, label), dim=1)

                (
                    output_dim,
                    loss_fct_name,
                    label_shape,
                    weight,
                ) = VISUAL_CONFIG.visual_loss_config[key]
                visn_loss_fct = loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(
                    visn_prediction_scores.view(-1, output_dim),
                    label.view(*label_shape),
                )
                if visn_loss.dim() > 1:  # Regression Losses
                    visn_loss = visn_loss.mean(1)

                # Only take the loss corresponding to features, not lang input - EXP
                visn_loss = visn_loss[label.size(0) * 128 :].clone()

                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
                losses += (visn_loss.detach(),)
            total_loss += total_visn_loss

        return total_loss, torch.stack(losses).unsqueeze(0), None
