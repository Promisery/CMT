import torch
import torch.nn as nn
from typing import Dict, Union


class PredictionModel(nn.Module):
    """
    Single-agent prediction model
    """
    def __init__(self, encoder,
                 aggregator,
                 decoder):
        """
        Initializes model for single-agent trajectory prediction
        """
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.decoder = decoder

    def forward(self, inputs: Dict) -> Union[torch.Tensor, Dict]:
        """
        Forward pass for prediction model
        :param inputs: Dictionary with
            'target_agent_representation': target agent history
            'surrounding_agent_representation': surrounding agent history
            'map_representation': HD map representation
        :return outputs: K Predicted trajectories and/or their probabilities
        """
        encodings = self.encoder(inputs)
        agg_encoding = self.aggregator(encodings)
        outputs = self.decoder(agg_encoding)

        return outputs
