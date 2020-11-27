import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(
            rng: np.random.Generator,
            data: pd.DataFrame
        ) -> None:
    """The code to recreate the plot from the paper.

    Args:
        data (pandas.DataFrame): The pandas DataFrame of results to plot.
    """
    # Remove training trials
    data = data[data['Main']]
    # Calculate mean Tp by condition
    mean_tp = data.groupby(['Cond', 'Ts'])['Tp'].mean().reset_index()
    # Determine axes limits
    yrange = np.multiply(
        (min(mean_tp['Ts']), max(mean_tp['Ts'])),
        [0.95, 1.05]
    )

    # Subset data for plotting
    cond1 = mean_tp.loc[mean_tp['Cond'] == 1]
    cond2 = mean_tp.loc[mean_tp['Cond'] == 2]
    cond3 = mean_tp.loc[mean_tp['Cond'] == 3]

    # Add jitter noise
    jitter = data.copy()
    jitter['Ts'] = jitter['Ts'] + rng.uniform(-5, 5, len(data))
    cond1_jitter = jitter.loc[jitter['Cond'] == 1]
    cond2_jitter = jitter.loc[jitter['Cond'] == 2]
    cond3_jitter = jitter.loc[jitter['Cond'] == 3]

    # Make plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(xlim=yrange, ylim=yrange)
    fig.gca().set_aspect('equal', adjustable='box')

    ax.set_xlabel('Sample interval (ms)')
    ax.set_ylabel('Production time (ms)')

    ax.plot(yrange, yrange, linestyle='--', color='gray')

    ax.scatter(
        cond1_jitter['Ts'], cond1_jitter['Tp'],
        marker='.', color='black', alpha=0.025, label=None
    )
    ax.scatter(
        cond2_jitter['Ts'], cond2_jitter['Tp'],
        marker='.', color='brown', alpha=0.025, label=None
    )
    ax.scatter(
        cond3_jitter['Ts'], cond3_jitter['Tp'],
        marker='.', color='red', alpha=0.025, label=None
    )

    ax.plot(
        cond1['Ts'], cond1['Tp'],
        color='black', marker='o', label="short"
    )
    ax.plot(
        cond2['Ts'], cond2['Tp'],
        color='brown', marker='o', label="intermediate"
    )
    ax.plot(
        cond3['Ts'], cond3['Tp'],
        color='red', marker='o', label="long"
    )

    ax.legend(title='Prior condition', loc=4)
    plt.show()
