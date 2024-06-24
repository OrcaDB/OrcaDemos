from typing import Any

import torch
import torch.nn as nn
from orcalib.orca_torch import OrcaLookupLayer, _LinearClassificationHead


def ListDictToDictList(list_dict: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Converts a list of dictionaries to a dictionary of lists
    """
    dict_list = {}
    for key in list_dict[0].keys():
        dict_list[key] = []
    for dict_ in list_dict:
        for key in dict_.keys():
            dict_list[key].append(dict_[key])
    return dict_list


def DictListToListDict(dict_list: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Converts a dictionary of lists to a list of dictionaries
    """
    list_dict = []
    for i in range(len(dict_list[list(dict_list.keys())[0]])):
        list_dict.append({})
    for key in dict_list.keys():
        for i in range(len(dict_list[key])):
            list_dict[i][key] = dict_list[key][i]
    return list_dict


class OrcaGenericCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        embedding_size,
        v_size,
        memory_index,
        memory_label,
        num_memories,
        dropout,
        activation,
    ):
        super().__init__()
        self.v_size = v_size
        self.embedding_size = embedding_size
        self.memory_index = memory_index
        self.memory_label = memory_label

        self._last_accessed_memories = None
        # self.sparse_block_size = 20

        self.lookup = OrcaLookupLayer(
            index_name=memory_index,
            lookup_column_names=memory_label,
            num_memories=num_memories,
        )

        self.cross_attn = nn.MultiheadAttention(
            embedding_size,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
            # vdim=v_size
            vdim=embedding_size,
        )

        self.attn_norm = nn.LayerNorm(embedding_size)

        self.activation = activation

        self.linear1 = nn.Linear(embedding_size, embedding_size * 4)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embedding_size * 4, embedding_size)
        self.dropout2 = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(embedding_size)

    def forward(self, x, ctx, labels, memory_key):
        # for now, lets squeeze out the sequence dim
        # x = x.squeeze(1)
        x = x[:, -1, :]
        ctx, meta = self.lookup(x)
        self._last_accessed_memories = meta
        # unpacked_meta = [meta[0][i][self.memory_label] for i in range(len(meta[0]))]
        # labels = (
        #    torch.Tensor(unpacked_meta).to(torch.int64).to(x.device)
        # )  # .squeeze(-1)

        # unsqueeze to match memory dim (this is NOT sequence dim)
        x = x.unsqueeze(1)  # N x 1 x D
        # values = F.one_hot(labels, self.v_size).to(x.dtype).to(x.device)

        # Create sparse mask
        # sparse_mask = torch.zeros(values.shape[1])
        # sparse_mask[torch.randperm(len(sparse_mask))[:self.sparse_block_size]] = 1

        x, _ = self.cross_attn(x, ctx, ctx)  # N x 1 x D
        # x, _ = self.cross_attn(x, ctx, values.unsqueeze(0))  # N x 1 x D
        # x = x * sparse_mask

        x = x.squeeze(1)  # N x D
        x = self.attn_norm(x)  # N x D

        y = self.linear1(x)  # N x D*4
        y = self.activation(y)
        y = self.dropout1(y)
        y = self.linear2(y)  # N x D
        y = self.dropout2(y)
        x = self.ff_norm(y + x)  # N x D

        # unsqueeze back the seq dimension
        x = x.unsqueeze(1)

        return x


class OrcaLLMHead(nn.Module):
    """
    This will be similar to OrcaClassificationHead but for generic language models
    """

    def __init__(
        self,
        embedding_size,
        vocab_size,
        memory_index,
        memory_label,
        classifier=None,
        activation=torch.nn.functional.relu,
        dropout=0,
    ):
        super().__init__()
        self.memory_layer = OrcaGenericCrossAttentionLayer(
            embedding_size=embedding_size,
            v_size=vocab_size,
            memory_index=memory_index,
            memory_label=memory_label,
            num_memories=20,
            dropout=dropout,
            activation=torch.nn.functional.relu,
        )

        if classifier is None:
            self.classifier = _LinearClassificationHead(
                embedding_size, vocab_size, dropout=dropout, activation=activation
            )
        else:
            # let's use the existing training head so we don't have to retrain a new one
            # we should be sure to freeze this for training
            # TODO cleaner way of inserting before this layer?
            self.classifier = classifier
        self._memory_enabled = True
        self.deep_residuals = False

    def forward(self, x, ctx=None, labels=None):
        if self._memory_enabled:
            mem_informed_x = self.memory_layer(x, ctx, labels, None)
        x = self.classifier(x + mem_informed_x)
        return x
