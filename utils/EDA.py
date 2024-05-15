import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import hiplot as hip
import plotly.express as px


# Definindo uma paleta de cores fixa
color_palette = [
    '#1f77b4',  # Azul
    '#bcbd22',  # Verde-oliva
    '#e377c2',  # Rosa
    '#2ca02c',  # Verde
    '#d62728',  # Vermelho
    '#9467bd',  # Roxo
    '#ff7f0e',  # Laranja
    '#8c564b',  # Marrom
    '#7f7f7f',  # Cinza
    '#17becf'   # Ciano-azulado
]

def plot_all_axis(df, process, machine, by_code = False, decimation_factor=100):

    """
    Plots data for X_axis, Y_axis, and Z_axis in separate subplots for each unique code or year-month combination,
    for a specified process and machine, with a fixed color scale for consistency.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - by_code: Boolean, true to group by unique code, false to group only by year and month.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)].copy()

    # Create a subplot for each axis
    fig = make_subplots(rows=3, cols=1, subplot_titles=('X_axis', 'Y_axis', 'Z_axis'))

    # Determine group by columns
    if by_code:
        # Use Unique_Code for coloring
        unique_codes = filtered_df['Unique_Code'].unique()
        color_mapping = {code: color_palette[i % len(color_palette)] for i, code in enumerate(unique_codes)}
        
        for i, code in enumerate(unique_codes):
            df_plot = filtered_df[filtered_df['Unique_Code'] == code][::decimation_factor]
            if not df_plot.empty:
                for j, axis in enumerate(['X_axis', 'Y_axis', 'Z_axis'], start=1):
                    fig.add_trace(go.Scatter(
                        x=df_plot['Time'],
                        y=df_plot[axis],  # Plot each axis
                        mode='markers',
                        name=f'Code: {code} - {axis}',
                        marker=dict(color=color_mapping[code])
                    ), row=j, col=1)
    else:
        # Use Period for coloring
        filtered_df['Period'] = filtered_df['Year'].astype(str) + '-' + filtered_df['Month']
        periods = filtered_df['Period'].unique()
        color_mapping = {period: color_palette[i % len(color_palette)] for i, period in enumerate(periods)}
        
        for i, period in enumerate(periods):
            df_plot = filtered_df[filtered_df['Period'] == period][::decimation_factor]
            if not df_plot.empty:
                for j, axis in enumerate(['X_axis', 'Y_axis', 'Z_axis'], start=1):
                    fig.add_trace(go.Scatter(
                        x=df_plot['Time'],
                        y=df_plot[axis],  # Plot each axis
                        mode='markers',
                        name=f'Period: {period} - {axis}',
                        marker=dict(color=color_mapping[period])
                    ), row=j, col=1)

    # Update layout of the plot
    fig.update_layout(
        title=f'{process} on Machine {machine}',
        xaxis_title='Time',
        yaxis_title='Axis Value',
        template='plotly_white',
        legend_title="Group",
        height=900  # Increase the height to accommodate three subplots
    )

    # Show the plot
    fig.show()

def plot_all_axis_matplotlib(df, process, machine, by_code=False, decimation_factor=100):
    """
    Plots data for X_axis, Y_axis, and Z_axis in separate subplots for each unique code or year-month combination,
    for a specified process and machine, with a fixed color scale for consistency using matplotlib.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - by_code: Boolean, true to group by unique code, false to group only by year and month.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)].copy()

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(3, 1, figsize=(12,10), sharex=True)

    # Titles for each subplot
    axis_titles = ['X_axis', 'Y_axis', 'Z_axis']

    # Determine group by columns
    groups = filtered_df['Period'].unique() 
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))  # Generate colors from a colormap

    for i, group in enumerate(groups):
        group_df = filtered_df[filtered_df['Period'] == group][::decimation_factor]
        for j, axis in enumerate(['X_axis', 'Y_axis', 'Z_axis']):
            axes[j].scatter(group_df['Time'], group_df[axis], label=f'{group} - {axis}', color=colors[i % len(colors)], s=10)
            axes[j].set_title(axis_titles[j])
            axes[j].set_xlabel('Time')
            axes[j].set_ylabel('Value')
            axes[j].legend(title="Period", loc='upper right',bbox_to_anchor=(1.1, 1.05))
            axes[j].grid(True)

    fig.suptitle(f'{process} on Machine {machine}', fontsize=18)
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(df, machine, process, sample_frac=0.05, random_state=0):
    """
    Plota uma matriz de dispersão para as colunas 'X_axis', 'Y_axis', e 'Z_axis' para um dado
    processo e máquina em um DataFrame.

    Parâmetros:
    - df: DataFrame contendo os dados.
    - machine: String representando a máquina a ser filtrada.
    - process: String representando o processo a ser filtrado.
    - sample_frac: Fração do DataFrame para amostragem (default 0.05).
    - random_state: Semente para a geração de números aleatórios (default 0).
    """

    # Filtrar dados por máquina e processo
    df_filtered = df[(df['Machine'] == machine) & (df['Process'] == process)]

    # Ordenar as categorias de período se a coluna 'Period' existe no DataFrame
    if 'Period' in df_filtered.columns:
        code_order = df_filtered['Period'].unique()
    else:
        code_order = None

    # Definir as colunas para a matriz de dispersão
    cols = ['X_axis', 'Y_axis', 'Z_axis']

    # Criar a matriz de dispersão
    fig = px.scatter_matrix(df_filtered.sample(frac=sample_frac, random_state=random_state),
                            dimensions=cols, color='Period',
                            category_orders={'Period': code_order} if code_order is not None else None)

    # Atualizar layout do gráfico
    fig.update_layout(width=1200, height=800, legend_title_font_size=22)

    # Atualizar características dos traços
    fig.update_traces(marker=dict(size=5), diagonal_visible=False, showupperhalf=False)

    # Exibir o gráfico
    fig.show()


def plot_by_code_index_matplotlib(df, process, machine, axis='Z_axis', decimation_factor=100):
    """
    Plots data for a specified axis for groups of unique codes using Matplotlib,
    with each subplot containing up to 5 unique codes, for a specified process and machine,
    with a manually defined color scale for consistency, and grid enabled for better visualization.
    
    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - axis: String, the axis to plot (e.g., 'X_axis', 'Y_axis', 'Z_axis').
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)]
    filtered_df.reset_index(drop=True, inplace=True)

    # Get unique codes
    unique_codes = filtered_df['Unique_Code'].unique()
    num_subplots = np.ceil(len(unique_codes) / 5).astype(int)

    # Calculate global y limits
    global_y_min = filtered_df[axis].min()
    global_y_max = filtered_df[axis].max()
    
    # Create the subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots), sharey=True)
    
    # Check if axs is iterable
    if not hasattr(axs, '__iter__'):
        axs = [axs]
    
    # Define a custom color palette
    color_palette = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if len(unique_codes) > len(color_palette):
        # Extend the palette with random colors if there are more unique codes than the default palette length
        extra_colors = np.random.rand(len(unique_codes) - len(color_palette), 3)
        color_palette.extend(extra_colors)

    # Plot each unique code in its corresponding subplot
    for i, ax in enumerate(axs):
        codes_to_plot = unique_codes[i*5:(i+1)*5]
        for idx, code in enumerate(codes_to_plot):
            df_plot = filtered_df[filtered_df['Unique_Code'] == code][::decimation_factor]
            if not df_plot.empty:
                ax.plot(df_plot.index, df_plot[axis], label=f'{code}', color=color_palette[idx % len(color_palette)])
        
        # Set the same y-axis limit for all subplots
        ax.set_ylim(global_y_min, global_y_max)
        
        # Set legend, title, and labels
        ax.legend(loc='lower right')
        ax.set_title(f'{process} on {machine} - {axis}')
        ax.set_xlabel('Index')
        ax.set_ylabel(axis)

        # Enable grid
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_selected_columns(df, process, machine, columns_to_plot, max_codes_per_plot=5, decimation_factor=100):
    """
    Plots specified columns with a maximum of 5 unique codes per plot.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - columns_to_plot: List of column names to be plotted.
    - max_codes_per_plot: int, maximum number of unique codes per plot.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)]
    filtered_df.reset_index(drop=True, inplace=True)

    # Get unique codes
    unique_codes = filtered_df['Unique_Code'].unique()

    # Ensure the columns to plot exist in the DataFrame
    available_columns = filtered_df.columns
    columns_to_plot = [col for col in columns_to_plot if col in available_columns]
    num_columns = len(columns_to_plot)

    # Calculate the number of plots needed per column
    num_plots = np.ceil(len(unique_codes) / max_codes_per_plot).astype(int)

    # Create subplots with one column per row
    fig, axs = plt.subplots(num_columns * num_plots, 1, figsize=(12, 4 * num_columns * num_plots), sharey=False)

    # Ensure axs is iterable
    if len(axs) == 1:
        axs = [axs]

    # Define a custom color palette
    color_palette = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if len(unique_codes) > len(color_palette):
        extra_colors = np.random.rand(len(unique_codes) - len(color_palette), 3)
        color_palette.extend(extra_colors)

    # Track subplot index
    subplot_idx = 0

    # Loop through columns and plots
    for col_idx, column in enumerate(columns_to_plot):
        for i in range(num_plots):
            ax = axs[subplot_idx]
            codes_to_plot = unique_codes[i * max_codes_per_plot:(i + 1) * max_codes_per_plot]

            # Get the y-limits for the current column
            global_y_min = filtered_df[column].min()
            global_y_max = filtered_df[column].max()

            for idx, code in enumerate(codes_to_plot):
                df_plot = filtered_df[filtered_df['Unique_Code'] == code][::decimation_factor]
                if not df_plot.empty:
                    ax.plot(df_plot.index, df_plot[column], label=f'{code}', color=color_palette[idx % len(color_palette)])
            
            # Set the y-axis limits
            ax.set_ylim(global_y_min, global_y_max)

            # Set legend, title, and labels
            ax.legend(loc='lower right')
            ax.set_title(f'{process} on {machine} - {column}')
            ax.set_xlabel('Index')
            ax.set_ylabel(column)

            # Enable grid
            ax.grid(True)

            # Increment subplot index for the next plot
            subplot_idx += 1

    # Adjust layout
    plt.tight_layout()
    plt.show()

def visualize_with_hiplot(df):
    """
    Visualizes a Pandas DataFrame using HiPlot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to visualize.
    
    Returns:
    - Displays an interactive HiPlot visualization.
    """
    # Convert the DataFrame to a HiPlot experiment
    data = df.to_dict(orient='records')
    exp = hip.Experiment.from_iterable(data)
    
    # Display the HiPlot visualization
    exp.display()

def plot_scatter_matrix_FE(df, machine, process, cols, sample_frac=0.05, random_state=0):
    """
    Plots a scatter matrix for specified columns for a given process and machine in a DataFrame,
    highlighting different 'Unique_Code' values.

    Parameters:
    - df: DataFrame containing the data.
    - machine: String representing the machine to filter by.
    - process: String representing the process to filter by.
    - cols: List of column names to include in the scatter matrix.
    - sample_frac: Fraction of the DataFrame to sample (default 0.05).
    - random_state: Seed for random number generation (default 0).
    """

    # Filter data by machine and process
    df_filtered = df[(df['Machine'] == machine) & (df['Process'] == process)]

    # Ensure only columns that exist in the DataFrame are used
    cols = [col for col in cols if col in df_filtered.columns]

    # Determine the column order for 'Unique_Code'
    unique_code_order = df_filtered['Unique_Code'].unique()

    # Create the scatter matrix
    fig = px.scatter_matrix(df_filtered.sample(frac=sample_frac, random_state=random_state),
                            dimensions=cols, color='Unique_Code',
                            category_orders={'Unique_Code': list(unique_code_order)})

    # Update layout
    fig.update_layout(width=1400, height=1000, legend_title_font_size=22)
  

    # Update trace characteristics
    fig.update_traces(marker=dict(size=5), diagonal_visible=False, showupperhalf=False)

    # Display the figure
    fig.show()

def plot_all_axis_matplotlib(df, process, machine, by_code=False, decimation_factor=100):
    """
    Plots data for X_axis, Y_axis, and Z_axis in separate subplots for each unique code or year-month combination,
    for a specified process and machine, with a fixed color scale for consistency using matplotlib.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - by_code: Boolean, true to group by unique code, false to group only by year and month.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)].copy()

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(3, 1, figsize=(12,12), sharex=True)

    # Titles for each subplot
    axis_titles = ['X_axis', 'Y_axis', 'Z_axis']

    # Determine group by columns
    groups = filtered_df['Period'].unique() 
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))  # Generate colors from a colormap

    for i, group in enumerate(groups):
        group_df = filtered_df[filtered_df['Period'] == group][::decimation_factor]
        for j, axis in enumerate(['X_axis', 'Y_axis', 'Z_axis']):
            axes[j].scatter(group_df['Time'], group_df[axis], label=f'{group} - {axis}', color=colors[i % len(colors)], s=10)
            axes[j].set_title(axis_titles[j])
            axes[j].set_xlabel('Time')
            axes[j].set_ylabel('Value')
            axes[j].legend(title="Period", loc='upper right',bbox_to_anchor=(1.1, 1.05), fontsize=12)
            axes[j].grid(True)

    fig.suptitle(f'{process} on Machine {machine}', fontsize=18)
    # Adjust layout
    plt.tight_layout()
    plt.show()


def plotly_scattermatrix(
        df, cols, category_order=None, symbol=None, color="Month", upload=False, filename='Scattermatrix',
        width=1800, height=1100, label_fontsize=16, legend_fontsize=14):
    """Create a scatter matrix plot using Plotly Express.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        cols (list[str]): The columns to plot in the scatter matrix.
        category_order (dict[str, list], optional): The order of categories for the color parameter. Defaults to None.
        symbol (str, optional): The column name to use as symbols. Defaults to None.
        color (str, optional): The column name to use for coloring the points. Defaults to "Month".
        upload (bool, optional): If True, the plot is uploaded to Plotly's online platform. Defaults to False.
        filename (str, optional): The filename for the uploaded plot. Defaults to 'Scattermatrix'.
        width (int, optional): The width of the plot in pixels. Defaults to 1800.
        height (int, optional): The height of the plot in pixels. Defaults to 1100.
        label_fontsize (int, optional): The font size of x and y axis labels. Defaults to 16.
        legend_fontsize (int, optional): The font size of the legend. Defaults to 14.

    Returns:
        None
    """

    # Create the scatter matrix plot
    fig = px.scatter_matrix(df, dimensions=cols, category_orders=category_order, color=color, symbol=symbol)

    # Update the layout of the plot
    fig.update_layout(
        title='Pairplot',
        width=width,
        height=height,
        hovermode='closest',
        font=dict(size=label_fontsize),
        legend=dict(font=dict(size=legend_fontsize))
    )

    # Update the trace properties
    fig.update_traces(showupperhalf=False,diagonal_visible=False, marker=dict(size=2.5))

    # Show the plot
    fig.show()

    return