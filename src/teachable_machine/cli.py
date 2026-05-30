"""Command-line interface (CLI) for the teachable_machine package."""

import os
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from teachable_machine import ImageClassifier, ModelPatcher, DatasetEvaluator
from teachable_machine.exceptions import TeachableException

# Detect if the console environment supports UTF-8
try:
    _encoding = (sys.stdout.encoding or "ascii").lower()
except Exception:
    _encoding = "ascii"

_supports_unicode = "utf" in _encoding or "65001" in _encoding

# Initialize console with safe box drawing if unicode is not supported
console = Console(safe_box=True if not _supports_unicode else None)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Teachable Machine CLI - Professional tools for inference, patching, and evaluation."""
    pass


@main.command()
@click.argument("image_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option(
    "-m",
    "--model",
    "model_path",
    default="model/keras_model.h5",
    show_default=True,
    help="Path to Keras H5 model file.",
)
@click.option(
    "-l",
    "--labels",
    "labels_path",
    default="model/labels.txt",
    show_default=True,
    help="Path to labels file.",
)
def predict(image_path: str, model_path: str, labels_path: str):
    """Classifies a single image and prints prediction results."""
    try:
        # Load and run classifier
        with console.status("[bold green]Loading model and classifier..."):
            classifier = ImageClassifier(model_path, labels_path)

        with console.status("[bold blue]Preprocessing and predicting..."):
            result = classifier.predict(image_path)

        # Print result beautifully
        panel_content = (
            f"[bold]Predicted Class:[/bold] [bold cyan]{result.class_name}[/bold cyan]\n"
            f"[bold]Confidence Score:[/bold] [bold green]{result.confidence * 100:.2f}%[/bold green]\n\n"
            f"[bold underline]Class Breakdown:[/bold underline]\n"
        )
        block_char = "█" if _supports_unicode else "="
        bg_char = "░" if _supports_unicode else "-"
        for name, prob in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
            bar = block_char * int(prob * 20) + bg_char * (20 - int(prob * 20))
            panel_content += f"  {name:<15} [{bar}] {prob * 100:.2f}%\n"

        console.print(
            Panel(
                panel_content,
                title="[bold yellow]Inference Result[/bold yellow]",
                border_style="cyan",
            )
        )

    except TeachableException as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected System Error:[/bold red] {e}")
        sys.exit(1)


@main.command()
@click.argument("model_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option(
    "--no-backup",
    is_flag=True,
    help="Modify H5 model file directly without creating a .bak copy.",
)
def patch(model_path: str, no_backup: bool):
    """Patches a legacy Teachable Machine Keras H5 model for compatibility with Keras 3."""
    create_backup = not no_backup
    try:
        console.print(f"[bold yellow]Patching model metadata for:[bold white] {model_path}...")
        modified = ModelPatcher.patch(model_path, create_backup=create_backup)

        if modified:
            ok_symbol = "✔" if _supports_unicode else "[OK]"
            console.print(f"[bold green]{ok_symbol} Model patched successfully for Keras 3 compatibility![/bold green]")
            if create_backup:
                console.print(f"[dim]Backup file created at: {model_path}.bak[/dim]")
        else:
            console.print("[yellow]! Model did not require patching (already clean).[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error patching model:[/bold red] {e}")
        sys.exit(1)


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "-m",
    "--model",
    "model_path",
    default="model/keras_model.h5",
    show_default=True,
    help="Path to Keras H5 model file.",
)
@click.option(
    "-l",
    "--labels",
    "labels_path",
    default="model/labels.txt",
    show_default=True,
    help="Path to labels file.",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to save the Markdown report.",
)
def evaluate(dataset_path: str, model_path: str, labels_path: str, output_file: str):
    """Evaluates classifier performance across a structured image dataset folder."""
    try:
        with console.status("[bold green]Loading classifier for evaluation..."):
            classifier = ImageClassifier(model_path, labels_path)
            evaluator = DatasetEvaluator(classifier)

        with console.status("[bold blue]Running bulk dataset inference..."):
            report = evaluator.evaluate(dataset_path)

        # Print summary
        console.print("\n[bold yellow]Dataset Evaluation Summary[/bold yellow]")
        console.print(f"Accuracy: [bold green]{report.accuracy * 100:.2f}%[/bold green]")
        console.print(f"Processed: [bold]{report.total_processed}[/bold] images")
        if report.total_failed > 0:
            console.print(f"Failed: [bold red]{report.total_failed}[/bold red] images")
        console.print("")

        # Draw Class Performance Table
        perf_table = Table(title="Class Performance", show_lines=True)
        perf_table.add_column("Class", style="cyan", no_wrap=True)
        perf_table.add_column("Precision", justify="right")
        perf_table.add_column("Recall", justify="right")
        perf_table.add_column("F1-Score", justify="right")
        perf_table.add_column("Support", justify="right", style="magenta")

        for name in report.all_classes:
            metrics = report.class_metrics[name]
            perf_table.add_row(
                name,
                f"{metrics.precision:.4f}",
                f"{metrics.recall:.4f}",
                f"{metrics.f1_score:.4f}",
                str(metrics.support),
            )
        console.print(perf_table)
        console.print("")

        # Draw Confusion Matrix Table
        cm_table = Table(title="Confusion Matrix (Row: Actual, Column: Predicted)", show_lines=True)
        cm_table.add_column("Actual \\ Predicted", style="cyan", no_wrap=True)
        for name in report.all_classes:
            cm_table.add_column(name, justify="right")

        for actual in report.all_classes:
            row_cells = [actual]
            for predicted in report.all_classes:
                val = report.confusion_matrix[actual].get(predicted, 0)
                # Highlight diagonal elements (correct predictions)
                if actual == predicted:
                    cell_val = f"[bold green]{val}[/bold green]" if val > 0 else "0"
                else:
                    cell_val = f"[red]{val}[/red]" if val > 0 else "0"
                row_cells.append(cell_val)
            cm_table.add_row(*row_cells)
        console.print(cm_table)

        # Write file if requested
        if output_file:
            md_content = report.render_markdown()
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(md_content)
            ok_symbol = "✔" if _supports_unicode else "[OK]"
            console.print(f"\n[green]{ok_symbol} Markdown report saved to: [bold white]{output_file}[/bold white][/green]")

    except TeachableException as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Evaluation Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
