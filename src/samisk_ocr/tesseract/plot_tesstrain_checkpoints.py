import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.colors
import plotly.graph_objects as go


def checkpoint_modelname_to_modelname(checkpoint_modelname: str) -> str:
    return checkpoint_modelname.split(".")[0].rsplit("_", maxsplit=1)[0]


def get_fig(model_name: str, df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for score in df.score.unique():
        for split in df.split.unique():
            sub_df = df[df.model_name == model_name]
            sub_df = sub_df.sort_values("iteration")
            sub_df = sub_df[sub_df.split == split]
            sub_df = sub_df[sub_df.score == score]
            fig.add_trace(
                go.Scatter(
                    x=sub_df["iteration"],
                    y=sub_df["value"],
                    mode="markers",
                    name=f"{model_name} {score} {split}",
                    marker=dict(color=color_map[f"{score}_{split}"]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sub_df["iteration"],
                    y=sub_df["value"],
                    mode="lines",
                    name=f"Line for {model_name}_{score}_{split}",
                    line=dict(width=1, color=color_map[f"{score}_{split}"]),
                    opacity=0.5,
                    showlegend=False,
                )
            )
    best_cer_iteration = (
        df[df.model_name == model_name][df.score == "CER"][df.split == "val"]
        .sort_values("value")
        .head(1)
        .iteration.item()
    )
    best_cer_iteration = "{:_}".format(best_cer_iteration)
    best_wer_iteration = (
        df[df.model_name == model_name][df.score == "WER"][df.split == "val"]
        .sort_values("value")
        .head(1)
        .iteration.item()
    )
    best_wer_iteration = "{:_}".format(best_wer_iteration)

    fig.update_layout(
        title=f"CER and WER scores on {model_name} training checkpoints <br>Best WER iteration: {best_wer_iteration}<br>Best CER iteration: {best_cer_iteration}",
        xaxis_title="Number of iterations",
        yaxis_title="Score",
    )
    return fig


if __name__ == "__main__":
    parser = ArgumentParser(
        "Plot eval results of tesstrain checkpoints, save with tesseract models"
    )
    parser.add_argument(
        "eval_output_dir",
        type=Path,
        help="Directory where output from eval_tesstrain_checkpoints is stored",
    )
    args = parser.parse_args()

    data_dict = defaultdict(list)

    for split in args.eval_output_dir.iterdir():
        for e in split.iterdir():
            score_dict = json.loads(e.read_text())
            model_name = checkpoint_modelname_to_modelname(score_dict["model_name"])

            for score in ("CER", "WER"):
                data_dict["split"].append(split.name)
                data_dict["model_name"].append(model_name)
                data_dict["iteration"].append(score_dict["iteration"])
                data_dict["score"].append(score)
                data_dict["value"].append(score_dict[score])

    df = pd.DataFrame(data_dict)

    # Create color map for plotting
    colors = plotly.colors.qualitative.Dark24
    color_map = {}
    i = 0
    for score in df.score.unique():
        for split in df.split.unique():
            color_map[f"{score}_{split}"] = colors[i]
            i += 1

    # Plot and save
    df = df.sort_values("model_name")
    for model_name in df.model_name.unique():
        fig = get_fig(model_name=model_name, df=df)
        model_stem = "".join([char for char in model_name if not char.isnumeric()])
        output_dir = Path(f"tesseract_models/{model_stem}")
        output_dir.mkdir(exist_ok=True)
        fig.write_image(output_dir / f"{model_name}_checkpoints.png")
