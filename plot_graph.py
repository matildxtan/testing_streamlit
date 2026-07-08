import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_growth_chart():

    if not os.path.exists(
        "results.csv"
    ):

        print(
            "results.csv not found"
        )

        return

    df = pd.read_csv(
        "results.csv"
    )

    if df.empty:

        print(
            "CSV is empty"
        )

        return

    df.columns = (
        df.columns
        .str.strip()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        df["Image"],
        df["Duckweed %"],
        marker="o",
        linewidth=2.5,
        label="Duckweed Coverage"
    )

    plt.title(
        "Duckweed Growth Trend"
    )

    plt.xlabel(
        "Images"
    )

    plt.ylabel(
        "Coverage (%)"
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "duckweed_growth_curve.png",
        dpi=300
    )

    plt.close()

    print(
        "Graph generated successfully"
    )


if __name__ == "__main__":

    generate_growth_chart()