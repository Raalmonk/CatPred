"""Display independently measured ESM and prediction stages from one run."""
import json
import math
from pathlib import Path
import statistics


def stage_rows(summary):
    if summary.get('status') != 'COMPLETE' or not summary.get('all_reported_timings_passed_exact_bits'):
        raise ValueError('Finish the exact-output comparison before plotting')
    esm = summary['preparation']['esm_comparison']
    if esm.get('status') != 'COMPLETE' or not esm.get('all_exact_bits'):
        raise ValueError('ESM outputs must match before plotting')
    stages = (
        ('ESM features + loading', esm['arms']['Original'], esm['arms']['Optimized']),
        ('Prediction (models loaded)', summary['arms']['Original'], summary['arms']['S_STREAM_K1']),
    )
    rows = []
    for label, original, optimized in stages:
        medians = []
        for arm in (original, optimized):
            times = arm['seconds']
            if not times or any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError('Stage timings must be finite and positive')
            median = statistics.median(times)
            if not math.isclose(median, arm['median_seconds'], rel_tol=1e-12, abs_tol=0):
                raise ValueError('Stage median differs from the measured repetitions')
            medians.append(median)
        rows.append({'Stage': label, 'Original (s)': medians[0], 'Optimized (s)': medians[1],
                     'Speedup': medians[0] / medians[1]})
    return rows


def plot_stages(summary, output):
    import matplotlib.pyplot as plt
    rows = stage_rows(summary)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'stage_comparison.json').write_text(json.dumps({'rows': rows}, indent=2) + '\n')
    colors = ['#66758A', '#167D9A']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
                         'axes.labelcolor': '#495669', 'xtick.color': '#495669', 'ytick.color': '#172638'}):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), layout='constrained')
        for axis, row in zip(axes, rows):
            values = [row['Original (s)'], row['Optimized (s)']]
            bars = axis.barh(['Original', 'Optimized'], values, height=0.48, color=colors)
            axis.invert_yaxis()
            axis.set_xlim(0, max(values) * 1.28)
            axis.set_xlabel('Seconds')
            axis.set_title(row['Stage'] + '\n' + f"{row['Speedup']:.2f}× speedup", loc='left', pad=14)
            axis.set_axisbelow(True)
            axis.grid(axis='x', color='#E5E9EF', linewidth=0.8)
            axis.tick_params(axis='both', length=0, pad=8)
            for spine in axis.spines.values():
                spine.set_visible(False)
            for bar, value in zip(bars, values):
                axis.text(value + max(values) * 0.025, bar.get_y() + bar.get_height() / 2,
                          f'{value:.3f} s', va='center', color='#172638', fontsize=11)
        fig.suptitle(f"{summary['rows']:,} rows, {summary['preparation']['unique_sequences']:,} unique proteins",
                     x=0.01, ha='left', fontsize=14, color='#172638')
        fig.savefig(output / 'stage_comparison.png', dpi=170, facecolor='white')
        fig.savefig(output / 'stage_comparison.svg', facecolor='white')
    return fig, rows
