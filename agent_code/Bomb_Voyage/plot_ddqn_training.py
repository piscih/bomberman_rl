
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_STAGE_NAMES = ("coin-heaven", "loot-crate", "classic")


def clean_columns(df):
    """Remove accidental markdown markers such as ** from column names."""
    df.columns = [str(c).replace("**", "").strip() for c in df.columns]
    return df


def check_chronological_order(df):
    """`total_steps_done` increments every step and is never reset, even
    across stage/process boundaries -- unlike `round`. If it isn't
    monotonically non-decreasing, the CSV isn't in the append-order we're
    assuming, and row-position-as-time-axis would be wrong too."""
    if "total_steps_done" not in df.columns:
        return
    s = pd.to_numeric(df["total_steps_done"], errors="coerce")
    if not s.is_monotonic_increasing:
        n_bad = (s.diff() < 0).sum()
        print(
            f"WARNING: total_steps_done decreases at {n_bad} row(s) -- "
            f"this CSV may not be in true chronological order. "
            f"Stage detection and the cumulative-round x-axis below assume "
            f"it is; treat the resulting plots with caution."
        )


def add_stage(df, stage_names=DEFAULT_STAGE_NAMES):
    """Split the log into stages by detecting where `round` resets
    (value drops instead of increasing), not by comparing its raw value
    to fixed thresholds -- see module docstring for why that's broken.
    Returns (df, boundary_positions) where boundary_positions are the
    row-positions of each detected reset, for drawing divider lines."""
    round_numeric = pd.to_numeric(df["round"], errors="coerce")
    reset_positions = list(np.flatnonzero(round_numeric.diff().to_numpy() <= 0))
    boundaries = [0] + reset_positions + [len(df)]

    n_segments = len(boundaries) - 1
    if n_segments != len(stage_names):
        print(
            f"WARNING: detected {n_segments} stage segment(s) via round-resets, "
            f"but {len(stage_names)} configured stage name(s) "
            f"({', '.join(stage_names)}). Labeling as many as line up; "
            f"pass --stage-names if your curriculum differs."
        )

    df["stage"] = pd.Series(pd.NA, index=df.index, dtype="string")
    for i in range(n_segments):
        start, end = boundaries[i], boundaries[i + 1]
        name = stage_names[i] if i < len(stage_names) else f"stage-{i + 1}"
        df.iloc[start:end, df.columns.get_loc("stage")] = name

    df["global_round"] = np.arange(1, len(df) + 1)
    return df, reset_positions


def rolling(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def add_stage_lines(ax, boundaries):
    for b in boundaries:
        ax.axvline(b, linestyle="--", linewidth=1, color="gray")


def make_plots(df, output_dir, window, boundaries):
    output_dir.mkdir(parents=True, exist_ok=True)
    x = df["global_round"]

    # ------------------------------------------------------------
    # 1. Reward
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, rolling(df["total_reward"], window), linewidth=1.5,
            label=f"{window}-round moving average")
    add_stage_lines(ax, boundaries)
    ax.set_xlabel("Training round (cumulative)")
    ax.set_ylabel("Total reward")
    ax.set_title("DDQN Training Reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reward_training.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # 2. Coins, crates, and kills
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, rolling(df["coins_collected"], window), linewidth=1.5, label="Coins collected")
    ax.plot(x, rolling(df["crates_destroyed"], window), linewidth=1.5, label="Crates destroyed")
    if "kills" in df.columns:
        ax.plot(x, rolling(df["kills"], window), linewidth=1.5, label="Opponents killed")
    add_stage_lines(ax, boundaries)
    ax.set_xlabel("Training round (cumulative)")
    ax.set_ylabel("Average per round")
    ax.set_title("Resource Collection & Kills During Training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "resources_training.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # 3. Suicide rate
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    suicide_rate = rolling(df["suicide"], window) * 100
    ax.plot(x, suicide_rate, linewidth=1.5, label=f"{window}-round moving average")
    add_stage_lines(ax, boundaries)
    ax.set_xlabel("Training round (cumulative)")
    ax.set_ylabel("Suicide rate (%)")
    ax.set_title("Suicide Rate During Training")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "suicide_rate_training.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # 4. Crates per bomb (only if a bomb-count column exists)
    # ------------------------------------------------------------
    bomb_column = None
    for candidate in ["bombs", "bombs_placed", "bombs_used", "bomb_placed"]:
        if candidate in df.columns:
            bomb_column = candidate
            break

    if bomb_column is not None:
        bombs = df[bomb_column].replace(0, pd.NA)
        crates_per_bomb = df["crates_destroyed"] / bombs

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, rolling(crates_per_bomb, window), linewidth=1.5,
                label=f"{window}-round moving average")
        add_stage_lines(ax, boundaries)
        ax.set_xlabel("Training round (cumulative)")
        ax.set_ylabel("Crates destroyed per bomb")
        ax.set_title("Bomb Efficiency During Training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "crates_per_bomb_training.png", dpi=300)
        plt.close(fig)
    else:
        print("Skipping crates_per_bomb_training.png: no bomb-count column was found.")

    # ------------------------------------------------------------
    # 5. Epsilon
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, df["epsilon"], linewidth=1.5)
    add_stage_lines(ax, boundaries)
    ax.set_xlabel("Training round (cumulative)")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_title("Exploration Rate During Training")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / "epsilon_training.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # 6. Average loss
    # ------------------------------------------------------------
    loss = pd.to_numeric(df["avg_loss"], errors="coerce")
    if loss.notna().any():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, rolling(loss, window), linewidth=1.5, label=f"{window}-round moving average")
        add_stage_lines(ax, boundaries)
        ax.set_xlabel("Training round (cumulative)")
        ax.set_ylabel("Average loss")
        ax.set_title("Training Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_training.png", dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------
    # 7. Survival rate (bonus: complements suicide rate directly)
    # ------------------------------------------------------------
    if "survived_round" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        survival_rate = rolling(df["survived_round"], window) * 100
        ax.plot(x, survival_rate, linewidth=1.5, label=f"{window}-round moving average")
        add_stage_lines(ax, boundaries)
        ax.set_xlabel("Training round (cumulative)")
        ax.set_ylabel("Survival rate (%)")
        ax.set_title("Round Survival Rate During Training")
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "survival_rate_training.png", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to the DDQN training CSV")
    parser.add_argument("--output", default="training_plots", help="Directory in which plots are saved")
    parser.add_argument("--window", type=int, default=100, help="Moving-average window in rounds (default: 100)")
    parser.add_argument(
        "--stage-names", nargs="+", default=list(DEFAULT_STAGE_NAMES),
        help="Names of curriculum stages in order (default: coin-heaven loot-crate classic)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_dir = Path(args.output)

    df = pd.read_csv(csv_path)
    df = clean_columns(df)

    numeric_columns = [
        "round", "steps", "total_steps_done", "epsilon", "total_reward",
        "avg_reward_per_step", "coins_collected", "crates_destroyed", "kills",
        "died", "suicide", "survived_round", "opponents_remaining",
        "buffer_size", "optimizer_updates", "avg_loss",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"round", "epsilon", "total_reward", "coins_collected", "crates_destroyed", "suicide", "avg_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # NOTE: deliberately NOT sorting by `round` -- see module docstring.
    # The file's own row order IS the chronological order.
    check_chronological_order(df)
    df, boundaries = add_stage(df, tuple(args.stage_names))

    make_plots(df, output_dir, args.window, boundaries)

    print(f"Loaded {len(df):,} training rounds.")
    print(f"Detected {len(boundaries)} stage boundary(ies) at cumulative-round positions: {boundaries}")
    print(f"Saved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()