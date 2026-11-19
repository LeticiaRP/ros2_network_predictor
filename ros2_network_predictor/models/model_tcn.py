import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from data_pipeline import HORIZONS, QUANTILES


# ---------- TCN training parameters -------------------
N_INPUTS = 6
N_HORIZONS = len(HORIZONS)
N_QUANTILES = len(QUANTILES)
N_OUTPUTS = N_HORIZONS * N_QUANTILES
N_CHANNELS = [32, 64, 128, 64]
KERNEL_SIZE = 3
DROPOUT = 0.2
STRIDE = 1


class Chomp1d(nn.Module):
    """
    Removes trailing elements (padding) from the end of a 1D temporal sequence to maintain
    casuality in convulutional networks. 
    """


    def __init__(self, chomp_size): 
        """
        Initializes the Chomp1d module

        Arguments: 
            - chomp_size: int, number of elements to remove from the end of the sequence
        """

        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size



    def forward(self, x): 
        """
        Slices off the trailing padding along the sequence dimension 

        Arguments: 
            - x: tensor of shape (batch_size, channels, seq_len) representing the padded sequence

        Returns: 
            - tensor: contiguous tensor of shape (batch size, channels, seq_len - chomp_size) 
        """

        return x[:, :, : - self.chomp_size].contiguous()
    



class TemporalBlock(nn.Module): 
    """
    Residual building block for a Temporal Convolutional Network (TCN) consisting of two dilated 
    causal convolutions and a skip connection 
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, 
                 stride, dropout, dilation, padding):
        """
        Initializes the TemporalBlock with two causal convolution layers and optimal 1x1 
        convolution for matching residual dimension

        Arguments: 
            - n_inputs: number of input channels
            - n_outputs: number of output channels 
            - kernel_size: size of the convolving kernel 
            - stride: stride of the convolution
            - dilation: spacing between kernel elements
            - padding: zero-padding added to both sides of the input before chomping 
            - dropout: dropout probability
        """

        super(TemporalBlock, self).__init__()

        # Passes the input through two identical causal convolution stages -------------

        self.conv1 = weight_norm(                                               # Applies a dilated 1D convolution with weight normalization 
            nn.Conv1d(                                                          # which decouples the lenght of weight vectors from their direction to         
                n_inputs, n_outputs, kernel_size,                               # stabilize and speed um traning 
                stride=stride, padding=padding, dilation=dilation               
            )
        )
        self.chomp1 = Chomp1d(padding)                                          # Slices off the extra right-side padding added by Conv1d so future
                                                                                # time steps never leak into the past 
        
        self.relu1 = nn.ReLU()                                                  # Adds non-linearity
        self.dropout1 = nn.Dropout(dropout)                                     # Adds regularization 


        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size, 
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)



        self.net = nn.Sequential(                                               
            self.conv1, 
            self.chomp1, 
            self.relu1, 
            self.dropout1, 

            self.conv2, 
            self.chomp2, 
            self.relu2, 
            self.dropout2
        )


        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()


    def forward(self, x): 
        """
        Passes the input through the dilated causal convolutions and adds the residual connection

        Arguments: 
            - x: torch. Tensor of shape (batch_size, n_inputs, seq_len)
        
        Returns
            - torch.Tensor: output tensor of shape (batch_size, n_outputs, seq_len)
        """

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)



class TCN(nn.Module): 
    """
    Temporal Convolutional Network (TCN) for multi-horizon probabilistic time-series 
    forecast using quantile regression 
    """

    def __init__(self, input_dim = N_INPUTS, num_horizons = N_HORIZONS, num_quantiles = N_QUANTILES, 
                 num_channels = N_CHANNELS, kernel_size = KERNEL_SIZE, dropout = DROPOUT, stride = STRIDE): 
        """
        Initializes the TCN with exponentially dilated TemporalBlocks and a multi-quantile projection ahead 

        Arguments: 
            - input_dim = number of input features per time step 
            - num_horizons = number of future time steps to forecast 
            - num_quantiles = number of quantiles predicted per horizons 
            - num_channels = hidden channel sizes for each TCN level
            - kernel_size = size of convolving kernel in each TemporalBlock 
            - dropout = dropout probability inside each TemporalBlock 
        """

        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i 
            in_channels = input_dim if i==0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs = in_channels, 
                    n_outputs = out_channels, 
                    kernel_size = kernel_size, 
                    stride = stride, 
                    dilation = dilation_size, 
                    padding = (kernel_size - 1) * dilation_size, 
                    dropout = dropout 
                )
            )

        self.network = nn.Sequential(*layers)

        self.num_horizons = num_horizons
        self.num_quantiles = num_quantiles
        self.linear = nn.Linear(num_channels[-1], self.num_horizons * self.num_quantiles)



    def forward(self, x): 
        """
        Generates multi-horizon quantile forecasts from an input historical window 

        Arguments: 
            - x: tensor of shape (batch_size, seq_len, input_dim) containing historical features 

        Returns: 
            - torch.Tensor: forecasts tensor of shape (batch_size, num_horizons, num_quantiles)
        """

        # rearrange to 1D convulations 
        x = x.permute(0, 2, 1)
        y = self.network(x)

        # extract the final causal time step (batch_size, num_channels[-1]) and project
        out = self.linear(y[:, :, -1])

        # reshape to (batch_size, num_horizons, num_quantiles)
        return out.view(out.shape[0], self.num_horizons, self.num_quantiles)





class LossFunction(nn.Module):
    """
    Horizon-weighted Pinball (Quantile) Loss for multi-horizon probabilistic forecasting. 
    Penalizes prediction errors asymmetrically per quantile and scales each horizon by inverse power weight h^(-beta)
    """


    def __init__(self, quantiles, horizons, beta): 
        """
        Initializes the Loss Function module and precomputes normalized horizon weights

        Arguments: 
            - quantiles: target quantile
            - horizons: forecast step offsets used to compute weights
            - beta: decay exponent controlling how strongly distant horizons are down-weightd 
        """

        super().__init__()

        self.quantiles = quantiles

        # compute inverse power weights w_h = h^(-beta) and normalize to sum to len(horizons)
        weights = torch.tensor([h ** (-beta) for h in horizons ], dtype=torch.float32)
        weights = weights / weights.sum() * len(horizons)
        self.register_buffer("horizon_weights", weights)




    def forward(self, preds, target): 
        """
        Computes the horizon-weighted quantile loss between predicted quantiles and ground truth

        Arguents: 
            - preds: torch.Tensor of shape
            - target: torch.Tensor of shape
        
        Return: 
            - torch.Tensor: scalar loss avareged across batch and horizons, summed over quantiles
        """

        loss = 0.0 
        target = target.unsqueeze(-1)

        for i, q in enumerate(self.quantiles): 
            preds_q = preds[:, :, i]    
            errors = target - preds_q.unsqueeze(-1)

            # assymetric pinball loss 
            pinball = torch.max((q-1)*errors, q*errors).squeeze(-1)

            # weight by horizon importance and average across batch and horizons
            if q == 0.95:
                q_weight = 10.0
            elif q == 0.50:
                q_weight = 2.0
            else:
                q_weight = 1.0
            
            loss += q_weight * (pinball * self.horizon_weights).mean()

        return loss 
