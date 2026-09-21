"""Plot a training run's loss from its log.

    python plot_loss.py                          # output/train_log.jsonl
    python plot_loss.py path/to/train.log        # the run folder's train.log works too

Each logged loss is one batch of 8 clips, or the mean of one accumulated
optimizer step when gradient accumulation is enabled, so the raw curve is
noisy. Read the moving average and the per-epoch means, not single points.
"""
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(HERE, 'output', 'train_log_seed1.jsonl')
SMOOTH = 20            # log points per moving-average window (20 x 50 = 1000 steps)
LABEL_SMOOTHING = 0.2  # Uni-Sign's default, see model_adapter.py
VOCAB = 250112         # mT5-base output size


def loss_floor(eps=LABEL_SMOOTHING, v=VOCAB):
    # The lowest loss label smoothing allows: the entropy of the smoothed
    # target, reached only when every token is predicted perfectly.
    hit = 1 - eps + eps / v
    return -hit * math.log(hit) - (v - 1) * (eps / v) * math.log(eps / v)


def read_log(path):
    # Accepts train_log.jsonl and train.log alike: keeps the {"step": ...} lines.
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('{"step"'):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # A resumed run can log a step twice; keep the latest.
    return sorted({r['step']: r for r in rows}.values(), key=lambda r: r['step'])


def moving_average(xs, n):
    out, total = [], 0.0
    for i, x in enumerate(xs):
        total += x
        if i >= n:
            total -= xs[i - n]
        out.append(total / min(i + 1, n))
    return out


def plot(path):
    if not os.path.exists(path):
        sys.exit(f'No log at {path}')
    rows = read_log(path)
    if not rows:
        sys.exit(f'{path} has no step lines yet')

    steps = [r['step'] for r in rows]
    losses = [r['loss'] for r in rows]
    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[r['epoch']].append(r['loss'])
    epochs = sorted(by_epoch)
    means = [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs]
    full = max(len(v) for v in by_epoch.values())
    partial = [len(by_epoch[e]) < 0.9 * full for e in epochs]
    epoch_starts = [next(r['step'] for r in rows if r['epoch'] == e) for e in epochs[1:]]
    floor = loss_floor()

    print(f'{len(rows)} log points, step {steps[0]}..{steps[-1]}')
    print('epoch  mean loss  points')
    for e, m, p in zip(epochs, means, partial):
        print(f'{e + 1:>5}  {m:9.3f}  {len(by_epoch[e]):>6}' + ('  (in progress)' if p else ''))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 10), gridspec_kw={'height_ratios': [3, 2, 1.2]})

    ax1.plot(steps, losses, color='tab:orange', alpha=0.25, lw=0.8, label='per batch (raw)')
    ax1.plot(steps, moving_average(losses, SMOOTH), color='tab:red', lw=2,
             label=f'moving average ({SMOOTH * (steps[1] - steps[0])} steps)')
    for s in epoch_starts:
        ax1.axvline(s, color='grey', lw=0.5, alpha=0.4)
    ax1.axhline(floor, color='black', ls='--', lw=1,
                label=f'floor with label smoothing {LABEL_SMOOTHING} ({floor:.2f})')
    ax1.set_title(f'{os.path.basename(path)}: loss by step (last step {steps[-1]}, '
                  f'epoch {rows[-1]["epoch"] + 1})')
    ax1.set_xlabel('step (grey lines = epoch boundaries)')
    ax1.set_ylabel('loss')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend()

    x = [e + 1 for e in epochs]       # the log counts from 0; show 1-based
    ax2.plot(x, means, marker='o', color='tab:blue')
    for xi, m, p in zip(x, means, partial):
        if p:
            ax2.plot(xi, m, marker='o', mfc='white', color='tab:blue', ms=9)
            ax2.annotate('in progress', (xi, m), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=8)
    ax2.set_title('mean loss per epoch')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('mean loss')
    ax2.set_xticks(x)
    ax2.grid(True, ls='--', alpha=0.5)

    ax3.plot(steps, [r['lr'] for r in rows], color='tab:green')
    ax3.set_title('learning rate')
    ax3.set_xlabel('step')
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax3.grid(True, ls='--', alpha=0.5)

    fig.tight_layout()
    png = os.path.splitext(path)[0] + '_loss.png'
    fig.savefig(png, dpi=120)
    print('saved', png)
    plt.show()


if __name__ == '__main__':
    plot(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG)
