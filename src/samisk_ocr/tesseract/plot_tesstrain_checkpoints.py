import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.colors
import plotly.graph_objects as go


def checkpoint_modelname_to_modelname(checkpoint_modelname: str) -> str:
    return checkpoint_modelname.split(".")[0].rsplit("_", maxsplit=1)[0]


def get_model_stem(model_name: str) -> str:
    return "".join([char for char in model_name if not char.isnumeric()])


def get_training_iteration(model_name: str, model_stem: str) -> int:
    return int(model_name[len(model_stem) :])


def get_best_iteration(df: pd.DataFrame, score: str, split: str) -> int:
    """Assumes df is already filtered on model name/stem"""
    return df[df.score == score][df.split == split].sort_values("value").head(1).iteration.item()


def get_best_score(df: pd.DataFrame, score: str, split: str) -> float:
    """Assumes df is already filtered on model name/stem"""
    return round(
        df[df.score == score][df.split == split].sort_values("value").head(1).value.item() * 100, 2
    )


def get_fig(model_name: str, df: pd.DataFrame, color_map: dict[str, str]) -> go.Figure:
    df = df[df.model_name == model_name]

    fig = go.Figure()

    for score in df.score.unique():
        for split in df.split.unique():
            sub_df = df[df.split == split]
            sub_df = sub_df[sub_df.score == score]
            sub_df = sub_df.sort_values("iteration")

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

    best_cer_iteration = get_best_iteration(df=df, score="CER", split="val")
    best_cer_iteration = "{:_}".format(best_cer_iteration)
    best_cer = get_best_score(df=df, score="CER", split="val")

    best_wer_iteration = get_best_iteration(df=df, score="WER", split="val")
    best_wer_iteration = "{:_}".format(best_wer_iteration)
    best_wer = get_best_score(df=df, score="WER", split="val")

    fig.update_layout(
        title=f"CER and WER scores on {model_name} training checkpoints <br>Best WER: {best_wer}%,  iteration: {best_wer_iteration}<br>Best CER: {best_cer}%, iteration: {best_cer_iteration}",
        xaxis_title="Number of iterations",
        yaxis_title="Score",
    )
    return fig


def get_overall_fig(model_stem: str, df: pd.DataFrame, color_map: dict[str, str]) -> go.Figure:
    df = df[df.model_stem == model_stem]
    best_iterations = {
        iteration: get_best_iteration(df=df_, score="CER", split="val")
        for iteration, df_ in df.groupby("training_iteration")
    }
    best_iterations[0] = 0
    df["total_iterations"] = df.apply(
        lambda row: row.iteration + best_iterations[row.training_iteration - 1], axis=1
    )

    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=1200,
        height=800,
    )
    for score in df.score.unique():
        for split in df.split.unique():
            for iteration in df.training_iteration.unique():
                sub_df = df[df.split == split]
                sub_df = sub_df[sub_df.score == score]
                sub_df = sub_df[sub_df.training_iteration == iteration]
                sub_df = sub_df.sort_values(["total_iterations"])

                x = sub_df.total_iterations
                x_labels = [f"{iteration}: {e}" for e in sub_df.iteration]

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        text=x_labels,
                        textposition="top center",
                        y=sub_df["value"],
                        mode="lines+markers",
                        name=f"{model_stem} {iteration} {score} {split}",
                        marker=dict(color=color_map[f"{score}_{split}_{iteration}"]),
                    )
                )

    best_cer = (
        df[df.model_stem == model_stem][df.score == "CER"][df.split == "val"]
        .sort_values("value")
        .head(1)
    )
    best_cer_training_iteration = best_cer.training_iteration.item()
    best_cer_training_iteration = "{:_}".format(best_cer_training_iteration)

    best_cer_iteration = best_cer.iteration.item()
    best_cer_iteration = "{:_}".format(best_cer_iteration)

    best_cer_score = get_best_score(df, score="CER", split="val")

    best_wer = (
        df[df.model_stem == model_stem][df.score == "WER"][df.split == "val"]
        .sort_values("value")
        .head(1)
    )
    best_wer_training_iteration = best_wer.training_iteration.item()
    best_wer_training_iteration = "{:_}".format(best_wer_training_iteration)

    best_wer_iteration = best_wer.iteration.item()
    best_wer_iteration = "{:_}".format(best_wer_iteration)

    best_wer_score = get_best_score(df, score="WER", split="val")

    fig.update_layout(
        title=f"CER and WER scores on {model_stem} training checkpoints <br>Best WER: {best_wer_score}%, training iteration: {best_wer_training_iteration}, checkpoint: {best_wer_iteration} <br>Best CER: {best_cer_score}%, training iteration: {best_cer_training_iteration}, checkpoint: {best_cer_iteration}",
        xaxis_title="Total number of iterations",
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
            model_stem = checkpoint_modelname_to_modelname(score_dict["model_name"])

            for score in ("CER", "WER"):
                data_dict["split"].append(split.name)
                data_dict["model_name"].append(model_stem)
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

    # Plot and save for each training iteration
    df = df.sort_values("model_name")
    for model_name in df.model_name.unique():
        fig = get_fig(model_name=model_name, df=df, color_map=color_map)
        model_stem = get_model_stem(model_name=model_name)
        output_dir = Path(f"tesseract_models/{model_stem}")
        output_dir.mkdir(exist_ok=True)
        fig.write_image(output_dir / f"{model_name}_checkpoints.png")

    # Plot and save across training iterations
    df["model_stem"] = df.model_name.apply(get_model_stem)
    df["training_iteration"] = df.apply(
        lambda row: get_training_iteration(model_name=row.model_name, model_stem=row.model_stem),
        axis=1,
    )

    # Create new color map
    i = 0
    color_map = {}
    for split in df.split.unique():
        for iteration in df.training_iteration.unique():
            color_map[f"WER_{split}_{iteration}"] = plotly.colors.qualitative.Dark2[i]
            color_map[f"CER_{split}_{iteration}"] = plotly.colors.qualitative.Set2[i]
            i += 1

    for model_stem, sub_df in df.groupby("model_stem"):
        fig = get_overall_fig(model_stem=model_stem, df=sub_df, color_map=color_map)

        output_dir = Path(f"tesseract_models/{model_stem}")
        fig.write_image(output_dir / f"{model_stem}_all_checkpoints.png")
