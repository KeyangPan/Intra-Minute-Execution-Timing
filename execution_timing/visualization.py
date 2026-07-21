# Section 1
# visualization functions used in feature_engineering

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np

# diverging map for correlation: red (-1) through neutral gray (0) to blue (+1)
CORR_CMAP = LinearSegmentedColormap.from_list('corr', ['#e34948', '#f0efec', '#2a78d6'])


def plot_time_series(df,
                     cols,
                     title,
                     color='#2a78d6'
                     ):
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 1.8 * len(cols)),
                             sharex=True, constrained_layout=True)
    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], color=color, linewidth=0.9)
        ax.set_title(col, fontsize=10, loc='left')
        ax.grid(axis='y', color='0.9', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle(f'{title} first tick per minute', fontsize=12, x=0.01, ha='left')
    plt.show()


def plot_corr_heatmap(df,
                      cols,
                      title):
    corr = df[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2, 0.9 * n))
    im = ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(n), cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n), cols, fontsize=9)
    for i in range(n):
        for j in range(n):
            value = corr.iloc[i, j]
            text_color = 'white' if abs(value) > 0.6 else '#3a3a38'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()

# Section 2
# visualization functions used in prediction_model

def plot_coefficients(coefs,
                      title='Model coefficients',
                      unit='log-odds per 1 SD of factor'):
    """
    Horizontal bar chart of regression coefficients (pd.Series indexed by
    factor name). With standardized features, magnitudes are comparable
    across factors. Positive bars in blue, negative in red, matching the
    correlation heatmap convention.
    """
    coefs = coefs.sort_values()
    colors = ['#2a78d6' if v >= 0 else '#e34948' for v in coefs]

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(coefs) + 1))
    ax.barh(coefs.index, coefs.values, color=colors, height=0.6)
    ax.axvline(0, color='0.4', linewidth=0.8)

    for name, value in coefs.items():
        ax.text(value, name, f' {value:.3f} ',
                va='center', ha='left' if value >= 0 else 'right',
                fontsize=8, color='#3a3a38')

    ax.set_xlabel(unit, fontsize=9)
    ax.margins(x=0.15)
    ax.grid(axis='x', color='0.9', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()


def plot_prob_by_class(y_true,
                       y_pred_prob,
                       title='Predicted probability by realized class',
                       bins=80):
    """
    Overlaid histograms of predicted probability, one per realized class.
    Each histogram is normalized to density so the shapes are comparable
    despite class imbalance. The horizontal separation between the two
    distributions is the visual counterpart of AUC.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(y_pred_prob[y_true == 1], bins=bins, density=True,
            color='#2a78d6', alpha=0.55, label='realized up (y=1)')
    ax.hist(y_pred_prob[y_true == 0], bins=bins, density=True,
            color='#e34948', alpha=0.55, label='realized down/flat (y=0)')

    ax.set_xlabel('predicted P(ask goes up)', fontsize=9)
    ax.set_ylabel('density', fontsize=9)
    ax.grid(axis='y', color='0.9', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()


def plot_cross_corr_heatmap(df,
                            row_cols,
                            col_cols,
                            title):
    cross_corr = df[row_cols + col_cols].corr().loc[row_cols, col_cols]
    fig, ax = plt.subplots(figsize=(0.9 * len(col_cols) + 2, 0.9 * len(row_cols)))
    im = ax.imshow(cross_corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(col_cols)), col_cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_cols)), row_cols, fontsize=9)
    for i in range(len(row_cols)):
        for j in range(len(col_cols)):
            value = cross_corr.iloc[i, j]
            text_color = 'white' if abs(value) > 0.6 else '#3a3a38'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()


def plot_roc(y_true,
             y_pred_prob,
             title='ROC curve',
             color='#2a78d6'):
    """
    ROC curve of a probabilistic prediction vs the no-skill baseline.
    The baseline (predicting a constant, e.g. the base rate, for every
    sample) has no ranking information, so its ROC is the diagonal and
    its AUC is exactly 0.5 regardless of the constant.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=color, linewidth=1.5,
            label=f'model (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='0.6', linewidth=1.2, linestyle='--',
            label='baseline, constant prob (AUC = 0.500)')

    ax.set_xlabel('False positive rate', fontsize=9)
    ax.set_ylabel('True positive rate', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(color='0.9', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()

def display_quantile_table(y_pred,
                           y_true,
                           num_quantiles= 10,):
    eval_df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})

    # qcut = quantile cut: bin by equal COUNT
    eval_df['quantile'] = pd.qcut(
        x=eval_df['y_pred'],  # values to bin
        q=num_quantiles,  # number of bins
        labels=False,  # return bin numbers 0-9 instead of interval objects
    )

    quantile_table = eval_df.groupby('quantile').apply(
        lambda g: pd.Series({
            'mean_y_pred': g['y_pred'].mean(),
            'mean_y_true': g['y_true'].mean(),
            'size': len(g),
        })
    )
    return quantile_table

def plot_volatility_model(y_pred,
                          y_true,
                          price_series,
                          resample_freq = '1min',
                          title = 'Vol model through the day'):

    pred_rs = y_pred.resample(resample_freq).mean()
    real_rs = y_true.resample(resample_freq).mean()
    px_rs = price_series.resample(resample_freq).first().reindex(pred_rs.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True,
                                   height_ratios=[1, 1.4], constrained_layout=True)

    ax1.plot(px_rs.index, px_rs, color='0.4', linewidth=0.9)
    ax1.set_ylabel('px ($)', fontsize=9)
    ax1.set_title('Price', fontsize=10, loc='left')

    ax2.plot(real_rs.index, real_rs, color='#2a78d6', linewidth=1.1,
             label='realized log-std')
    ax2.plot(pred_rs.index, pred_rs, color='#e34948', linewidth=1.1,
             label='predicted log-std')
    ax2.set_ylabel(f'log-std ({resample_freq} mean)', fontsize=9)
    ax2.set_title('Volatility: predicted vs realized', fontsize=10, loc='left')
    ax2.legend(fontsize=8, frameon=False, loc='upper right')

    # default hourly date format is '%d %H:%M' (day-of-month first, e.g.
    # "01 10:00"); show time of day only, in the data's own timezone
    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter('%H:%M', tz=pred_rs.index.tz))

    for ax in (ax1, ax2):
        ax.grid(axis='y', color='0.9', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12, x=0.01, ha='left')
    plt.show()

# Section 3
# visualization functions used in strategy_backtest

def plot_cumulative_pnl(series_cum_pnl,
                        title='Cumulative PnL'):
    """
    Cumulative per-share PnL over time
    """

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(series_cum_pnl.index, series_cum_pnl, color='#2a78d6', linewidth=1.1)
    ax.axhline(0, color='0.6', linewidth=0.8, linestyle='--')

    ax.set_ylabel('cumulative pnl ($, 1 share per minute)', fontsize=9)
    ax.grid(axis='y', color='0.9', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()


def plot_execution_second_distribution(series_exec_seconds,
                                       title='Execution time within the minute',
                                       bins=60):
    """
    Histogram of 'execution_second' (seconds from minute start to fill).
    Mass near 60s = mostly deadline fills (no trigger before minute end);
    mass near 0s = signal is firing early in the minute.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(series_exec_seconds, bins=bins,
            color='#2a78d6', alpha=0.75)

    ax.set_xlabel('seconds from minute start', fontsize=9)
    ax.set_ylabel('count', fontsize=9)
    ax.grid(axis='y', color='0.9', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()


def plot_pnl_histogram(series_pnl,
                       title='PnL distribution',
                       bins=40):
    """
    Histogram of per-minute pnl. Mass on both sides of zero is expected;
    what matters is whether the right tail (wins) outweighs the left
    tail (losses) enough to make cumulative pnl positive despite a
    win rate that can sit below 0.5.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(series_pnl, bins=bins, color='#2a78d6', alpha=0.75)
    ax.axvline(0, color='0.4', linewidth=0.8, linestyle='--')
    ax.axvline(series_pnl.mean(), color='#e34948', linewidth=1.2,
              label=f'mean = {series_pnl.mean():.4f}')

    ax.set_xlabel('pnl ($/share)', fontsize=9)
    ax.set_ylabel('count', fontsize=9)
    ax.grid(axis='y', color='0.9', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()


def plot_price_path(sim_result,
                    title='Arrival bid/ask vs execution price'):
    """
    Per-minute arrival bid, arrival ask, and the fill (execution) price,
    all plotted against the minute index. Fill price bouncing between
    the arrival bid/ask band is the expected range; outside it signals
    the market moved before/around execution.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sim_result.index, sim_result['arrival_ask'], color='#2a78d6',
            linewidth=1.0, label='arrival ask')
    ax.plot(sim_result.index, sim_result['arrival_bid'], color='#e34948',
            linewidth=1.0, label='arrival bid')
    ax.plot(sim_result['execution_time'], sim_result['fill_price'], color='0.2',
            linewidth=1.0, linestyle='--', label='execution (fill) price')

    ax.set_ylabel('price ($)', fontsize=9)
    ax.grid(axis='y', color='0.9', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, loc='left')
    plt.show()


def plot_price_and_pnl(sim_result,
                       title='Arrival ask vs per-minute pnl'):
    """
    Arrival ask (left axis, line) and per-minute pnl (right axis, bars)
    overlaid on one shared time axis, so price level and pnl outcome can
    be compared directly without scanning between two panels. Positive pnl
    bars are a vivid green (wins should stand out); negative pnl bars are
    muted gray, so losses recede rather than compete visually with wins.
    """
    # bar width in days (matplotlib's date unit): ~0.8x the typical spacing
    # between rows, so bars don't touch when rows are exactly 1 minute apart
    step_seconds = pd.Series(sim_result.index).diff().dt.total_seconds().median()
    bar_width = (step_seconds / 86400) * 0.8

    fig, ax_price = plt.subplots(figsize=(12, 4.5))
    ax_pnl = ax_price.twinx()

    colors = ['#2ca989' if v >= 0 else '0.75' for v in sim_result['pnl']]
    ax_pnl.bar(sim_result.index, sim_result['pnl'], color=colors,
              width=bar_width, alpha=0.6, zorder=1)
    ax_pnl.axhline(0, color='0.5', linewidth=0.8, zorder=1)
    ax_pnl.set_ylabel('pnl ($/share)', fontsize=9)
    ax_pnl.tick_params(labelsize=8)

    # keep price lines on top of the pnl bars (twinx axes stack in creation
    # order by default, so ax_pnl -- created second -- would otherwise cover
    # ax_price's lines)
    ax_price.set_zorder(ax_pnl.get_zorder() + 1)
    ax_price.patch.set_visible(False)

    ax_price.plot(sim_result.index, sim_result['arrival_ask'], color='#2a78d6',
                 linewidth=1.2, label='arrival ask', zorder=3)
    ax_price.set_ylabel('price ($)', fontsize=9)
    ax_price.legend(fontsize=8, frameon=False, loc='upper left')

    ax_price.grid(axis='y', color='0.9', linewidth=0.5)
    ax_price.spines['top'].set_visible(False)
    ax_pnl.spines['top'].set_visible(False)
    ax_price.tick_params(labelsize=8)
    ax_price.set_title(title, fontsize=11, loc='left')
    plt.show()


def plot_coefficient_distribution(model_coeff,
                                  title='Coefficient stability across windows',
                                  unit='log-odds per 1 std of factor',
                                  bins=10):
    """
    One subplot per factor: histogram of its fitted coefficient across
    rolling windows. model_coeff: DataFrame, index = factor name, columns
    = window (as returned by run_simulation's second return value).

    A wide spread or sign flips across windows flags an unstable factor --
    a single fit's coefficient (plot_coefficients) can look confident while
    still being noise if it isn't consistent window to window.
    """
    order = model_coeff.mean(axis=1).sort_values(ascending=False).index

    fig, axes = plt.subplots(len(order), 1, figsize=(6, 1.6 * len(order)),
                             constrained_layout=True)
    for ax, factor in zip(axes, order):
        values = model_coeff.loc[factor].dropna()
        ax.hist(values, bins=bins, color='#2a78d6', alpha=0.75)
        ax.axvline(0, color='0.4', linewidth=0.8, linestyle='--')
        ax.set_title(factor, fontsize=10, loc='left')
        ax.grid(axis='y', color='0.9', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel(unit, fontsize=9)
    fig.suptitle(title, fontsize=12, x=0.01, ha='left')
    plt.show()