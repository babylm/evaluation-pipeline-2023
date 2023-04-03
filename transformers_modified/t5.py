import torch
import copy
from torch import nn
from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


def mean_pooling(inputs, mask):
    token_embeddings = inputs
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class MeanPooler(nn.Module):
    """ Calcualte simple average of the inputs """
    def __init__(self, input_size=None):
        super().__init__()

    def forward(self, inputs, mask=None):
        if mask is None:
            pooled_output = inputs.mean(dim=1)
        else:
            pooled_output = mean_pooling(inputs, mask)
        return None, pooled_output
 

class AdaptivePooler(nn.Module):
    """ Calcualte weighted average of the inputs with learnable weights """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.w = nn.Linear(self.input_size, 1, bias=True)

    def forward(self, inputs, mask=None):
        batch_size, seq_len, emb_dim = inputs.shape
        scores = torch.squeeze(self.w(inputs), dim=-1)
        weights = nn.functional.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdims=True)
        outputs = (inputs.permute(2, 0, 1) * weights).sum(-1).T
        return weights, outputs
 

class T5ForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config, pooler='adaptive'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = T5Stack(encoder_config, self.shared)

        pooler_class = AdaptivePooler if pooler == 'adaptive' else MeanPooler
        self.pooler = pooler_class(input_size=config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        weights, pooled_output = self.pooler(outputs[0], mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )