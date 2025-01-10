import torch
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


aggregation_rules = {
    'self_attn': ['self_attn.in_proj_weight', 'self_attn.in_proj_bias', 'self_attn.out_proj.weight', 'self_attn.out_proj.bias'],
    'multihead_attn': ['multihead_attn.in_proj_weight', 'multihead_attn.in_proj_bias', 'multihead_attn.out_proj.weight', 'multihead_attn.out_proj.bias'],
    'norm': ['norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'norm3.weight', 'norm3.bias'],
    'linear': ['linear1.w.weight', 'linear1.w.bias', 'linear1.v.weight', 'linear1.v.bias', 'linear2.weight', 'linear2.bias']
}

def get_layers_data(model, epoch=0, step=0):
    # Extract histograms
    layer_histograms = {'epoch': epoch, 'step': step}

    for name, param in model.named_parameters():
        if param.requires_grad or ('weight' in name or 'bias' in name):
            matched = False
            # Aggregate layer names based on rules
            for key, substrings in aggregation_rules.items():
                if any(sub in name for sub in substrings):
                    # If a match is found in the aggregation rule, format the name correctly
                    new_name = '.'.join(name.split('.')[:3]) + f'.{key}'
                    if 'weight' in name:
                        new_name += '.weight'
                    elif 'bias' in name:
                        new_name += '.bias'

                    # Now handle the data (calculating histograms, percentiles, etc.)
                    weights = param.data
                    sorted_weights = torch.sort(weights.flatten())[0]

                    # Calculate the 1st and 99th percentiles
                    lower_percentile = sorted_weights[int(0.01 * len(sorted_weights))]
                    upper_percentile = sorted_weights[int(0.99 * len(sorted_weights))]

                    # Efficient filtering using torch.clamp to limit values
                    filtered_weights = torch.clamp(weights, min=lower_percentile, max=upper_percentile)

                    # Compute the histogram (10 bins) for filtered weights using torch.histc
                    bin_counts = torch.histc(filtered_weights, bins=10, min=lower_percentile.item(), max=upper_percentile.item())

                    # Store histogram data
                    bin_edges = torch.linspace(lower_percentile, upper_percentile, steps=11).tolist()
                    layer_histograms[new_name] = {"bin_edges": bin_edges, "bin_counts": bin_counts.tolist()}
                    matched = True
                    break  # We found the matching rule, no need to check further rules

            # If no rule matched, use the original name and handle 'weight' and 'bias' correctly
            if not matched:
                new_name = name
                weights = param.data

                # Sort the tensor to calculate percentiles
                sorted_weights = torch.sort(weights.flatten())[0]

                # Calculate the 1st and 99th percentiles
                lower_percentile = sorted_weights[int(0.01 * len(sorted_weights))]
                upper_percentile = sorted_weights[int(0.99 * len(sorted_weights))]

                # Efficient filtering using torch.clamp to limit values
                filtered_weights = torch.clamp(weights, min=lower_percentile, max=upper_percentile)

                # Compute the histogram (10 bins) for filtered weights using torch.histc
                bin_counts = torch.histc(filtered_weights, bins=10, min=lower_percentile.item(), max=upper_percentile.item())

                # Store histogram data
                bin_edges = torch.linspace(lower_percentile, upper_percentile, steps=11).tolist()
                layer_histograms[new_name] = {"bin_edges": bin_edges, "bin_counts": bin_counts.tolist()}

    return layer_histograms



def save_layer_data(data, file_path):
    with open(file_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def plot_layers_data(layers_data):
    max_subplots_per_row = 4

    num_layers = len(layers_data[0])
    num_rows = (num_layers // max_subplots_per_row) + (1 if num_layers % max_subplots_per_row != 0 else 0)
    num_cols = max_subplots_per_row

    # Create a figure with subplots for each layer
    fig = make_subplots(rows=num_rows, cols=num_cols, shared_xaxes=False,
                        vertical_spacing=0.04, horizontal_spacing=0.05, subplot_titles=list([k.replace('.weight', '<br>.weight').replace('.bias', '<br>.bias') for k in layers_data[0].keys()]))

    # Iterate through the layers and plot the data
    for idx, (layer_name, data) in enumerate(layers_data[0].items()):
        row = idx // max_subplots_per_row + 1
        col = idx % max_subplots_per_row + 1

        for n_epoch, epoch_data in enumerate(layers_data):
            epoch_layer_data = epoch_data[layer_name]
            fig.add_trace(go.Scatter(
                x=epoch_layer_data["bin_edges"][:-1],
                y=[v + max(epoch_layer_data["bin_counts"])*n_epoch*0.1 for v in epoch_layer_data["bin_counts"]],
                name=layer_name,
                legendgroup=layer_name,  # Add this line
                showlegend=True if n_epoch == 0 else False,  
                fill='tozeroy',
                line=dict(color='darkorange'),
                opacity=0.5,
                offsetgroup=n_epoch,
            ), row=row, col=col)

    fig.update_layout(
        height=200 * num_rows,
        width=1200,
        template="plotly_dark",
        showlegend=True,
        annotations=[dict(font=dict(size=12)) for _ in fig['layout']['annotations']]
    )


    fig.show()
