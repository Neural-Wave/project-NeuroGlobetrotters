import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot
#scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from lingam import DirectLiNGAM
import plotly.graph_objects as go
import tkinter as tk

#DATA FOR TRAINING THE MODEL
X_Df = pd.read_csv('/Users/jonatan/Documents/hackathon/low_scrap.csv')
scaler = StandardScaler()
X = scaler.fit_transform(X_Df)

def backtrack_to_target(adjacency_matrix, columns, target_col='Station5_mp_85'):
    """
    Efficiently find all nodes with paths leading to the target node in a directed graph using backtracking.
    """
    # Convert columns to a list if it's an Index
    columns = list(columns)
    target_idx = columns.index(target_col)
    
    # Initialize the directed graph and set for nodes to backtrack
    G = nx.DiGraph()
    nodes_to_visit = [target_idx]
    visited_nodes = set()

    # Backtrack from target node through incoming edges
    while nodes_to_visit:
        current = nodes_to_visit.pop()
        if current not in visited_nodes:
            visited_nodes.add(current)
            for i, has_edge in enumerate(adjacency_matrix[:, current]):
                if has_edge != 0:  # Edge exists from i to current
                    G.add_edge(i, current, weight=has_edge)
                    nodes_to_visit.append(i)

    # Add node labels
    nx.set_node_attributes(G, {i: columns[i] for i in G.nodes()}, 'label')

    return G

def prior_knowledge_matrix(columns):
    """
    prior knowledge matrix for LiNGAM where:
    0: no directed path possible (temporal constraint violation)
    1: directed path 
    -1: no prior knowledge (we'll allow the algorithm to determine)
    """
    n_features = len(columns)
    prior_knowledge = np.full((n_features, n_features), -1)
    
    # get station number 
    def get_station_number(col_name):
        return int(col_name.split('_')[0].replace('Station', ''))
    
    # get measurement point number
    def get_mp_number(col_name):
        return int(col_name.split('_')[2])
    
    for i in range(n_features):
        for j in range(n_features):
            station_i = get_station_number(columns[i])
            station_j = get_station_number(columns[j])
            
            # constraint
            if station_i > station_j:
                prior_knowledge[i, j] = 0
            
            # should we allow internal dependencies? 
            # if station_i == station_j:
            #     prior_knowledge[i, j] = -1  # Let LiNGAM determine
    
    # No self loop allowed
    np.fill_diagonal(prior_knowledge, 0)
    
    return prior_knowledge


def plot_enhanced_subgraph(G, columns, path_effects, target_col='Station5_mp_85', save_path="fig1.png"):
    """
    Creates visualization with adaptive size and save functionality
    
    Parameters:
    - G: NetworkX graph
    - columns: DataFrame columns
    - path_effects: Dictionary of path effects
    - target_col: Target column name
    - save_path: Path to save figure (optional, e.g., 'causal_graph.png')
    """
    # Get screen size
    try:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        # Fallback to common resolution if can't get screen size
        screen_width = 1920
        screen_height = 1080
    
    # Calculate figure size based on screen resolution
    fig_width = screen_width / 100  # Convert pixels to inches
    fig_height = screen_height / 100
    
    # Setup
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)  # Lower DPI for display
    
    # Get most influential path
    most_influential_path = max(path_effects, key=path_effects.get)
    max_effect = path_effects[most_influential_path]
    
    # Create path edges for coloring
    influential_edges = list(zip(most_influential_path[:-1], most_influential_path[1:]))
    
    # Use hierarchical layout with more spacing
    pos = nx.spring_layout(G, k=4, iterations=100)  # Increased spacing
    
    # Draw regular edges
    regular_edges = [(u, v) for u, v in G.edges() 
                    if (u, v) not in influential_edges]
    
    nx.draw_networkx_edges(G, pos,
                          edgelist=regular_edges,
                          edge_color='gray',
                          width=1.5,
                          alpha=0.3,
                          arrows=True,
                          arrowsize=20)
    
    # Draw influential path edges
    nx.draw_networkx_edges(G, pos,
                          edgelist=influential_edges,
                          edge_color='red',
                          width=3,
                          alpha=0.8,
                          arrows=True,
                          arrowsize=25)
    
    # Draw nodes
    labels = nx.get_node_attributes(G, 'label')
    target_node = [n for n, d in G.nodes(data=True) if d['label'] == target_col][0]
    
    # Scale node sizes based on screen resolution
    base_size = min(screen_width, screen_height) / 10
    
    # Node colors and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in most_influential_path:
            if node == target_node:
                node_colors.append('#FF6B6B')
                node_sizes.append(base_size * 1.5)
            else:
                node_colors.append('#FFB347')
                node_sizes.append(base_size * 1.25)
        else:
            node_colors.append('#4ECDC4')
            node_sizes.append(base_size)
    
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.7,
                          linewidths=2,
                          edgecolors='white')
    
    # Calculate font size based on screen resolution
    font_size = min(screen_width, screen_height) / 100
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels,
                          font_size=font_size,
                          font_weight='bold',
                          font_color='black')
    
    # Add title
    plt.title(f"Causal Paths to Target Variable\nMost Influential Path Total Effect: {max_effect:.3f}", 
              pad=20, size=font_size*2, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor='#FF6B6B', markersize=15,
                   label='Target Variable'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#FFB347', markersize=15,
                   label='Most Influential Path Nodes'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#4ECDC4', markersize=15,
                   label='Other Causal Variables'),
        plt.Line2D([0], [0], color='red', lw=3,
                   label=f'Strongest Effect Path ({max_effect:.3f})')
    ]
    
    plt.legend(handles=legend_elements, 
              loc='upper right',
              fontsize=font_size,
              frameon=True)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save high-resolution figure if path provided
    # if save_path:
    #     # Create a new figure with higher DPI for saving
    #     save_fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    #     plt.savefig(save_path, 
    #                bbox_inches='tight', 
    #                dpi=300,
    #                facecolor='white',
    #                edgecolor='none',
    #                pad_inches=0.5)
    #     plt.close(save_fig)
    
    return plt

def plot_static_graph(adj_df, path_effects, target_label, title="Graph Visualization", 
                     target_color='#FF6B6B', default_color='#4ECDC4',
                     path_color='#FFB347'):
    """
    Creates a high-quality graph visualization with highlighted maximum effect path
    
    Parameters:
    adj_df: pandas DataFrame adjacency matrix
    path_effects: dictionary of paths and their effects
    target_label: string label of target node
    title: string for graph title
    target_color: color for target node
    default_color: color for other nodes
    path_color: color for maximum effect path nodes
    """
    # Get screen size
    try:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        # Fallback to common resolution if can't get screen size
        screen_width = 1920
        screen_height = 1080
    
    # Calculate figure size based on screen resolution
    fig_width = screen_width / 100  # Convert pixels to inches
    fig_height = screen_height / 100
    

    # Set style
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")
    
    # Get most influential path
    most_influential_path = max(path_effects, key=path_effects.get)
    max_effect = path_effects[most_influential_path]
    
    # Convert DataFrame to numpy array
    adj_matrix = adj_df.to_numpy()
    
    # Get node labels from DataFrame
    node_labels = dict(enumerate(adj_df.columns))
    
    # Get target node index
    target_node = adj_df.columns.get_loc(target_label)
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Set up the figure
    plt.figure(figsize=(fig_width, fig_height), dpi=100)
    #plt.figure(figsize=(20, 20), dpi=300)
    
    # Use better layout
    pos = nx.spring_layout(G, k=2, iterations=100)
    
    # Create path edges for coloring
    influential_edges = list(zip(most_influential_path[:-1], most_influential_path[1:]))
    
    # Draw regular edges
    regular_edges = [(u, v) for u, v in G.edges() 
                    if (u, v) not in influential_edges]
    
    # Draw regular edges
    nx.draw_networkx_edges(G, pos, 
                          edgelist=regular_edges,
                          width=1,
                          alpha=0.2, 
                          edge_color='#2C3E50',
                          style='solid')
    
    # Draw influential path edges
    nx.draw_networkx_edges(G, pos,
                          edgelist=influential_edges,
                          edge_color='red',
                          width=3,
                          alpha=0.8,
                          arrows=True,
                          arrowsize=25)
    
    # Calculate node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [3000 * (degrees[node] + 1) for node in G.nodes()]
    
    # Prepare node colors with path highlighting
    node_colors = []
    for i in G.nodes():
        if i == target_node:
            node_colors.append(target_color)
        elif i in most_influential_path:
            node_colors.append(path_color)
        else:
            node_colors.append(default_color)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.7,
                          linewidths=2,
                          edgecolors='white')
    
    # Draw labels with different styles
    target_labels = {node: label for node, label in node_labels.items() 
                    if node == target_node}
    path_labels = {node: label for node, label in node_labels.items() 
                  if node in most_influential_path and node != target_node}
    other_labels = {node: label for node, label in node_labels.items() 
                   if node not in most_influential_path and node != target_node}
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, target_labels,
                          font_size=14,
                          font_weight='bold',
                          font_color='black')
    
    nx.draw_networkx_labels(G, pos, path_labels,
                          font_size=12,
                          font_weight='bold',
                          font_color='#2C3E50')
    
    nx.draw_networkx_labels(G, pos, other_labels,
                          font_size=10,
                          font_color='#2C3E50')
    
    # Add title with effect information
    plt.title(f"{title}\nMaximum Total Effect: {max_effect:.3f}", 
              pad=20, size=20, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=target_color, markersize=15,
                   label='Target Variable'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=path_color, markersize=15,
                   label='Most Influential Path'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=default_color, markersize=15,
                   label='Other Variables'),
        plt.Line2D([0], [0], color='red', lw=3,
                   label=f'Strongest Effect ({max_effect:.3f})')
    ]
    
    plt.legend(handles=legend_elements, 
              loc='upper left',
              fontsize=12,
              frameon=True,
              facecolor='white',
              edgecolor='none',
              bbox_to_anchor=(0.02, 0.98))
    
    # Add info text
    plt.text(0.02, 0.02, 
             'Node size indicates number of connections\nEdge thickness indicates strength of relationship\nRed path shows strongest total effect',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def plot_interactive_graph(adj_df, path_effects, target_label, title="Graph Visualization"):
    """
    Creates a large interactive graph visualization with well-spaced nodes
    """
    # Get most influential path
    most_influential_path = max(path_effects, key=path_effects.get)
    max_effect = path_effects[most_influential_path]
    
    # Create graph
    G = nx.from_numpy_array(adj_df.to_numpy())
    target_node = adj_df.columns.get_loc(target_label)
    
    # Calculate layout with much more spacing
    pos = nx.spring_layout(G, k=5, iterations=100)  # Increased k for more spacing
    
    # Scale up the positions to spread nodes further
    for node in pos:
        pos[node] = pos[node] * 2  # Multiply positions by 2 for more spread
    
    # Prepare edge traces
    edge_trace_regular = []
    edge_trace_influential = []
    
    # Create path edges
    influential_edges = list(zip(most_influential_path[:-1], most_influential_path[1:]))
    
    # Create regular edges trace
    for edge in G.edges():
        if edge not in influential_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace_regular.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1.5, color='#2C3E50'),
                    opacity=0.2,
                    hoverinfo='none'
                )
            )
    
    # Create influential edges trace
    for edge in influential_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace_influential.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=4, color='red'),
                opacity=0.8,
                hoverinfo='none'
            )
        )
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    hover_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node colors
        if node == target_node:
            node_colors.append('#FF6B6B')
        elif node in most_influential_path:
            node_colors.append('#FFB347')
        else:
            node_colors.append('#4ECDC4')
            
        # Node sizes
        node_sizes.append(25 * (G.degree(node) + 1))
        
        # Node labels and hover text
        label = adj_df.columns[node]
        node_text.append(label)
        if node == target_node:
            hover_text.append(f"<b>Target:</b> {label}<br>Degree: {G.degree(node)}")
        elif node in most_influential_path:
            hover_text.append(f"<b>Path Node:</b> {label}<br>Degree: {G.degree(node)}")
        else:
            hover_text.append(f"<b>Node:</b> {label}<br>Degree: {G.degree(node)}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        hovertext=hover_text,
        textposition="bottom center",
        textfont=dict(size=12),
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line_width=2,
            line_color='white'
        )
    )
    
    # Create much larger figure
    fig = go.Figure(
        data=[*edge_trace_regular, *edge_trace_influential, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"{title}<br>Maximum Total Effect: {max_effect:.3f}",
                font=dict(size=24),
                y=0.95
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=100),
            annotations=[
                dict(
                    text="Node size indicates number of connections<br>Red path shows strongest total effect",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(size=14),
                    align="left"
                )
            ],
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[min(node_x)*1.2, max(node_x)*1.2]  # Increase axis range
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[min(node_y)*1.2, max(node_y)*1.2]  # Increase axis range
            ),
            plot_bgcolor='white',
            width=2000,  # Much larger width
            height=1500  # Much larger height
        )
    )
    
    # Add legend
    legend_items = [
        ("Target Variable", '#FF6B6B'),
        ("Most Influential Path", '#FFB347'),
        ("Other Variables", '#4ECDC4'),
        (f"Strongest Effect ({max_effect:.3f})", 'red')
    ]
    
    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=1.02,
            y=1 - (i * 0.1),
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=16),
            align="left",
            bgcolor="white",
            bordercolor=color,
            borderwidth=2,
            borderpad=4,
            opacity=0.8
        )
    
    # Update layout for better spacing
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        clickmode='event+select'
    )
    
    return fig



print("Calculating prior knowledge matrix...")
# Create the prior knowledge matrix
prior_knowledge = prior_knowledge_matrix(X_Df.columns)
prior_knowledge

print("Doing casual discovery")
model = DirectLiNGAM(prior_knowledge=prior_knowledge.T)
model.fit(X)
make_dot(model.adjacency_matrix_, labels=list(X_Df.columns)).render('constrained_graph_test', format='pdf')
adjusted_adjacency_matrix_df = pd.DataFrame(model.adjacency_matrix_, columns=X_Df.columns, index=X_Df.columns)
adjusted_adjacency_matrix_df.to_csv('./output/adj_matrix.csv')

print("Constructing subgraph")
G =  backtrack_to_target(model.adjacency_matrix_.T, X_Df.columns)

print("Calculating path effects")
target_idx = list(X_Df.columns).index("Station5_mp_85")
target_idx
paths_to_target = []
for node in G.nodes:
    if node != target_idx and nx.has_path(G, node, target_idx):
        paths_to_target.extend(nx.all_simple_paths(G, source=node, target=target_idx))
path_effects = {}
for path in paths_to_target:
    total_effect = 0
    for i in range(len(path) - 1):
        effect = model.estimate_total_effect(X_Df, path[i], path[-1])
        total_effect += effect
    path_effects[tuple(path)] = total_effect
print("plotting:")
# Plot
plot = plot_enhanced_subgraph(G, X_Df.columns,path_effects)
plot.show()
plot.savefig('./output/subgraph.png')

# Usage:
plot = plot_static_graph(adjusted_adjacency_matrix_df, path_effects, 'Station5_mp_85', 
                        title="Manufacturing Process Causal Graph")
plot.show()
plot.savefig('./output/complete.png')
# Usage
fig = plot_interactive_graph(adjusted_adjacency_matrix_df, path_effects, 'Station5_mp_85', 
                           title="Manufacturing Process Causal Graph")
#Optionally save
fig.write_html("causal_graph.html", include_plotlyjs=True)

print("Heriarchy of most influential path:")
print(path_effects.sort_values(ascending=False))