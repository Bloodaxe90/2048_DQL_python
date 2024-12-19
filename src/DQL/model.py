import torch
from torch import nn


class CustomConv2D(nn.Module):
    def __init__(self, input_neurons: int, output_neurons: int, state_size: int):
        super().__init__()
        divided_output_neurons = output_neurons // state_size
        kernel_sizes = [i+1 for i in range(state_size )]

        self.custom_conv2d = nn.ModuleList([
            nn.Conv2d(input_neurons, divided_output_neurons, kernel_size, padding="same")
            for kernel_size in kernel_sizes
        ])

    def forward(self, x) -> torch.Tensor:
        return torch.cat([conv2d(x) for conv2d in self.custom_conv2d], dim=1)

class DQCNN(nn.Module):

    def __init__(self, input_neurons: int, hidden_neurons: tuple, output_neurons: int, state_size: int, dropout: float = 0.5):
        super().__init__()

        self.input_block = nn.Sequential(
            CustomConv2D(input_neurons= input_neurons,
                      output_neurons= hidden_neurons[0],
                         state_size= state_size),
            nn.ReLU(),
        )


        self.hidden_blocks = nn.ModuleList()
        for i in range(1, len(hidden_neurons)):
            self.hidden_blocks.append(nn.Sequential(
               CustomConv2D(input_neurons= hidden_neurons[i-1],
                      output_neurons= hidden_neurons[i],
                            state_size= state_size),
               nn.ReLU(),
            ))

        flattened_in_neurons = hidden_neurons[-1]*(state_size**2)
        flattened_out_neurons = hidden_neurons[-1] //2
        self.output_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_in_neurons,
                      out_features=flattened_out_neurons),
            nn.Dropout(dropout),
            nn.Linear(in_features=flattened_out_neurons,
                      out_features=output_neurons)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_block(x)
        for hidden_block in self.hidden_blocks:
            x = hidden_block(x)
        return self.output_block(x)