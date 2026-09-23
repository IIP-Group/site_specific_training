"""Presentation and path helpers for the DUIDD training tutorial."""

from ast import literal_eval
from pathlib import Path

from IPython.display import Markdown, display


class NotebookHelpers:

    def __init__(self, workspace_root, display_root):
        self.workspace_root = Path(workspace_root).resolve()
        self.display_root = Path(display_root)

    def resolve(self, path):
        """Resolve a path relative to the workspace root."""
        path = Path(path)
        return path.resolve() if path.is_absolute() else (self.workspace_root / path).resolve()

    def display_path(self, path):
        """Return a workspace path with the concise root used in the tutorial."""
        relative_path = self.resolve(path).relative_to(self.workspace_root)
        return self.display_root / relative_path

    def show_terminal(self, command, outputs=()):
        """Render highlighted shell commands followed by plain terminal output."""
        command_lines = "\n".join(f"$ {line}" for line in command.splitlines())
        blocks = [f"```bash\n{command_lines}\n```"]
        if outputs:
            output_lines = "\n".join(f"{label}: {value}" for label, value in outputs)
            blocks.append(f"```text\n{output_lines}\n```")
        display(Markdown("\n\n".join(blocks)))

    def config_value(self, path, config_key):
        """Read one Python-literal value from a DUIDD configuration file."""
        resolved_path = self.resolve(path)
        for raw_line in resolved_path.read_text().splitlines():
            setting = raw_line.split("#", 1)[0].strip()
            key, separator, value = setting.partition("=")
            if separator and key.strip() == config_key:
                return literal_eval(value.strip())
        raise ValueError(f"No {config_key} found in {self.display_path(resolved_path)}")

    def show_config(self, path):
        """Print the configuration fields relevant to this tutorial."""
        resolved_path = self.resolve(path)
        relevant_keys = (
            "label =",
            "n_size_bwp =",
            "mcs_index =",
            "symbol_allocation =",
            "num_layers =",
            "n_rntis =",
            "n_ids =",
            "chest =",
            "duidd_schedule =",
            "num_iter_train_save =",
            "channel_type =",
            "datalake_tf_fn =",
        )
        print(f"--- {self.display_path(resolved_path)}\n")
        for line in resolved_path.read_text().splitlines():
            if line.strip().startswith(relevant_keys):
                print(line)
        print()
