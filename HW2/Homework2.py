from pathlib import Path
from typing import List
import sys; sys.path.append(Path().resolve().parent.as_posix())  # noqa

from ACTR.model import Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncexpon


rng = np.random.default_rng(seed=111)


def init_model(
            id: int
        ) -> Model:
    """We create a (model of a) participant here.

    This gives us a way to initialize them with some prior knowledge if
    necessary, and give them a unique id.

    Args:
        id (int): The unique id of the participant.

    Returns:
        Model: A fresh subject to experiment on.
    """
    subj = Model()
    subj.id = id
    return subj


def do_trial(
            subj: Model,
            sample_interval: float
        ) -> float:
    """This function takes a subject and an interval time, and goes through
    the process of an experiment trial.

    Args:
        subj (Model): The subject which will do the trial.
        sample_interval (float): The interval time defining the trial.

    Returns:
        float: The (re)production time.
    """
    # "Trials began with the presentation of a central fixation point for 1s,"
    subj.time += 1
    # "followed by the presentation of a warning stimulus ..."
    # "After a variable delay ranging from 0.25-0.85s drawn randomly from a
    # truncated exponential distribution, ..."
    subj.time += truncexpon(0.6, 0.25).rvs(1, random_state=rng)
    # "two 100ms flashes separated by the sample interval were presented."
    subj.time += 0.1  # READY
    subj.time += sample_interval
    # "Production times, t_p, were measured from the center of the flash, (that
    # is, 50ms after its onset) to when the key was pressed"
    subj.time += 0.05  # SET

    # implement cognitive model here
    production_time = sample_interval

    subj.time += production_time  # GO
    return production_time


def do_experiment(
            participants: List[Model],
            n_test_trials: int,
            n_train_trials: int
        ) -> pd.DataFrame:
    """Perform the experiment on a list of participants.

    Args:
        participants (list(Model)): The volunteering participants.
        n_test_trials (int): The number of testing trials.
        n_train_trials (int): The number of training trials.

    Returns:
        pandas.DataFrame: The collected data as a pandas DataFrame.
    """

    # "All priors were discrete uniform distributions with 11 values, ranging
    # from 494–847 ms for the short, 671–1,023 ms for the intermediate and
    # 847–1,200 ms for long prior condition."
    prior_distributions = np.array([
        (1, np.linspace(start=0.494, stop=0.847, num=11)),
        (2, np.linspace(start=0.671, stop=1.023, num=11)),
        (3, np.linspace(start=0.847, stop=1.200, num=11)),
    ], dtype='object')

    # create the DataFrame
    row_idx = 0
    data = pd.DataFrame(columns=[
        'Subj', 'Cond', 'line', 'Trial', 'Ts', 'Tp', 'MaxTrial', 'Main'
    ])

    # each subject goes through the experiment
    for subj in participants:

        # "For each subject, the order in which the three prior conditions were
        # tested was randomized."
        rng.shuffle(prior_distributions)
        for condition, distribution in prior_distributions:

            # "For each prior condition, subjects were tested after they
            # completed an initial learning stage." (here: train, then test)
            for trial_idx in range(1, n_train_trials + n_test_trials + 1):

                # "sample intervals are drawn randomly from one of three
                # partially overlapping discrete uniform prior distributions."
                sample_interval = rng.choice(distribution)

                # perform the actual trial
                production_time = do_trial(subj, sample_interval)
                # don't give feedback

                # record the results just like the authors did
                data.loc[row_idx] = [
                    subj.id + 1,
                    condition,
                    row_idx + 1,
                    trial_idx,
                    sample_interval,
                    production_time,
                    n_train_trials + n_test_trials,
                    False if trial_idx <= n_train_trials else True
                ]
                row_idx += 1

                # Reset declarative memory

            # Add one day delay after each block of trials

    return data


def plot(
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


def main(
            n_participants: int = 5,
            n_train_trials: int = 500,
            n_test_trials: int = 1000
        ) -> None:
    # create the participants, do the experiment and record the data
    participants = [init_model(id) for id in range(n_participants)]
    simulated_data = do_experiment(participants, n_test_trials, n_train_trials)
    simulated_data.to_csv(Path() / 'HW2' / 'model_data.csv', index=False)

    # load the original data, and plot both to compare
    original_data = pd.read_csv(Path() / "dataJS.csv")
    plot(original_data)
    plot(simulated_data)


if __name__ == '__main__':
    main()
