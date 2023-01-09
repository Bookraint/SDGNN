from random import betavariate
from re import S
import re
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import dim, unsqueeze
from transformers import BertModel, XLNetModel
from .bert import BertPreTrainedModel



class BertSent_Encode(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertSent_Encode, self).__init__(bert_config)
        self.bert_config = bert_config
        self.bert = BertModel(bert_config)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        return outputs