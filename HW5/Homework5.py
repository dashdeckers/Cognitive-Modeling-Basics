from pathlib import Path

import sys; sys.path.append(Path().resolve().parent.as_posix())  # noqa

from ACTR.model import Model
from ACTR.dmchunk import Chunk

import matplotlib.pyplot as plt
from scipy.stats import rv_discrete, uniform
import pandas as pd
import numpy as np

rng = np.random.default_rng(seed=111)


def noise(s: float) -> float:
    rand = rng.uniform(0.001, 0.999)
    return s * np.log((1 - rand) / rand)


def time_to_pulses(
            time: float,
            t_0: float = 0.011,
            a: float = 1.1,
            b: float = 0.015
        ) -> int:
    pulses = 0
    pulse_duration = t_0
    while time >= pulse_duration:
        time = time - pulse_duration
        pulses += 1
        pulse_duration = a * pulse_duration + noise(b * a * pulse_duration)
    return pulses


def pulses_to_time(
            pulses: int,
            t_0: float = 0.011,
            a: float = 1.1,
            b: float = 0.015
        ) -> float:
    time = 0
    pulse_duration = t_0
    while pulses > 0:
        time = time + pulse_duration
        pulses = pulses - 1
        pulse_duration = a * pulse_duration + noise(b * a * pulse_duration)
    return time


# TODO: type!?
def do_trial(subj: Model, distribution: str) -> float:
    # TODO:
    # subjs take breaks,
    # 'dist for run is reflected exactly, then shuffled'
    # calculate feedback and record it somewhere for the 0.8 thing

    # participants click the mouse after (?) to start the trial
    # time of their choosing is interpreted here as a uniform time in 0.5-3s
    subj.time += sum(rng.uniform(0.500, 3.000, size=1)) # click

    # x ms delay (sampled from distribution), then the yellow dot was flashed
    sample_interval = distribution.rvs(1) / 1000
    subj.time += sample_interval # flash



    # convert time to pulses and remember how many it took
    pulses = time_to_pulses(sample_interval)
    subj.add_encounter(Chunk(
        name=f'pf_{pulses}',
        slots={'isa': 'pulse-fact', 'pulses': pulses}
    ))


    # flash is shown for 18.5ms
    subj.time += 0.0185 # end flash


    # retrieve a blended trace
    request = Chunk(
        name='pulse-request',
        slots={'isa': 'pulse-fact'}
    )
    value, _ = subj.retrieve_blended_trace(request, 'pulses')
    # convert pulses back to time
    response_interval = pulses_to_time(value)



    # subjects were required to wait at least 250ms before reproducing
    # 'at least' is interpreted here as a uniform distribution from 250-350ms
    # TODO: i guess it would actually be more skewed towards 250ms
    subj.time += sum(rng.uniform(0.250, 0.350, size=1)) # end wait

    # interval is estimated and reproduced by holding the mouse click
    subj.time += response_interval # release mouse click

    # feedback is represented after a uniformly sampled delay (450-850ms)
    subj.time += sum(rng.uniform(0.450, 0.850, size=1)) # end delay

    # feedback is presented for 62ms
    # here, the feedback is not cognitively modeled
    subj.time += 0.062 # end feedback

    # fixation cross disappears after 500-750ms, followed by a blank screen for
    # another 500-750ms and then the trial restarts. interpreted as uniform.
    subj.time += sum(rng.uniform(0.500, 0.750, size=2))

    return sample_interval, response_interval


def get_error(interval, response, k=400, err_type='skewed'):
    # Get the error / feedback
    if err_type == 'skewed':
        return k * ( (response - interval) / response )

    if err_type == 'standard':
        return k * (response - interval)

    if err_type == 'fractional':
        return k * ( (response - interval) / interval )


def plot(spec: dict, data: pd.DataFrame) -> None:
    # Consider a single randomly chosen subject
    single = data[data['subject'] == rng.choice(pd.unique(data['subject']))]

    # Create and populate subplots
    fig, ((s_mean, g_mean), (s_std, g_std)) = plt.subplots(2, 2)

    subplot(s_mean, single, 'mean')
    s_mean.set_title('Single subject')
    s_mean.set_ylabel('Response bias (ms)')

    subplot(g_mean, data, 'mean')
    g_mean.set_title('Group mean')

    subplot(s_std, single, 'std')
    s_std.set_xlabel('Physical time interval (ms)')
    s_std.set_ylabel('Response sd (ms)')

    subplot(g_std, data, 'std')
    g_std.set_xlabel('Physical time interval (ms)')

    fig.suptitle(spec['name'])
    fig.tight_layout()
    fig.show()


def subplot(ax, data: pd.DataFrame, plot_type: str) -> None:
    # Extract and sort the interval values
    xticks = pd.unique(data['interval']) * 1000
    xticks.sort()

    # Group by distribution first to separate conditions
    for dist_val, dist_group in data.groupby('distribution'):

        # Group by interval and get mean reponse bias and s.e.m
        yvals = []
        yerr = []
        for int_val, int_group in dist_group.groupby('interval'):
            # Compute yval based on plot_type
            # plot_type is assumed to be a pandas method name
            val = getattr((int_group['response'] * 1000), plot_type)()
            if plot_type == 'mean': val -= int_val * 1000
            yvals.append(val)

            # Compute the s.e.m
            yerr.append((int_group['response'] * 1000).sem())

        # Slice the first 6 or the last 6 xticks depending on last int_val
        start, stop = (None, 6) if int_val * 1000 == xticks[5] else (-6, None)
        # Plot the points and the errorbars
        ax.errorbar(
            xticks[start:stop], yvals, yerr=yerr, label=dist_val,
            marker='o', markeredgewidth=1, markeredgecolor='black',
            linewidth=0, elinewidth=1, ecolor='black', capsize=1,
        )

    # Configure the plot
    if plot_type == 'mean': ax.set_ylim(-150, 150)
    if plot_type == 'std': ax.set_ylim(40, 120)
    ax.set_xlim(min(xticks) - 50, max(xticks) + 50)
    ax.hlines(0, *ax.get_xlim(), colors='gray', ls='-', lw=1)


def do_experiment(spec: dict) -> pd.DataFrame:
    data = list()

    # Subject loop
    for subj_id, subj in [(id, Model()) for id in range(spec['n_subj'])]:

        # Block loop
        rng.shuffle(spec['blocks'])
        for distribution in spec['blocks']:

            # Session / Trial loops (2 training sessions, 2 testing sessions)
            for sess_idx in range(4):
                for trial_idx in range(500):

                    # Do trial
                    interval, response = do_trial(subj, distribution)

                    # Collect data (ignore training sessions)
                    if sess_idx < 2: continue
                    data.append([
                        interval, response, get_error(interval, response),
                        distribution.name, subj_id,
                        sess_idx, trial_idx
                    ])

                # Between session break
                subj.time += 30

            # Between block break
            subj.time += 60  # between block break, could be higher

    return pd.DataFrame(data=data, columns=[
        'interval', 'response', 'feedback',
        'distribution', 'subject',
        'session_idx', 'trial_idx',
    ])


# Define Experiment1
exp1 = {
    # Unfortunately, scipy's custom discrete distributions only accept integer
    # value ranges, so these are defined in terms of ms and not s as units.
    'name': 'Experiment 1',
    'n_subj': 4,
    'blocks': np.array([
        rv_discrete(
            name='short',
            values=(
                np.linspace(start=450, stop=825, num=6),
                np.repeat(1/6, 6),
            ),
            seed=rng,
        ),
        rv_discrete(
            name='long',
            values=(
                np.linspace(start=750, stop=1125, num=6),
                np.repeat(1/6, 6),
            ),
            seed=rng,
        )
    ])
}

# Define Experiment2
exp2 = {
    'name': 'Experiment 2',
    'n_subj': 6,
    'blocks': np.array([
        rv_discrete(
            name='medium',
            values=(
                np.linspace(start=600, stop=975, num=6),
                np.repeat(1/6, 6),
            ),
            seed=rng,
        ),
        rv_discrete(
            name='medium_peaked',
            values=(
                np.linspace(start=600, stop=975, num=6),
                np.array([1/12, 7/12, 1/12, 1/12, 1/12, 1/12]),
            ),
            seed=rng,
        )
    ])
}


def main():
    data = do_experiment(exp1)
    plot(exp1, data)
    return data
