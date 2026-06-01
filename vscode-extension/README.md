# CodeScope VS Code Extension

Minimal VS Code extension prototype for running CodeScope from inside VS Code.

This v0 extension supports:

- `CodeScope: Index Workspace`
- `CodeScope: Investigate Bug`
- Running `python -m codescope.cli investigate <workspace> "<description>" --json`
- Logging full stdout and stderr to a `CodeScope` output channel
- Showing investigation results in the `CodeScope Results` TreeView
- Clicking result items to open files at the reported line

The extension calls the local CodeScope CLI. It does not call `--llm`, does not generate patches, and does not modify user files.

## Development Setup

Install extension dependencies:

```bash
npm install
```

Compile TypeScript:

```bash
npm run compile
```

Launch the Extension Development Host:

1. Open the `vscode-extension/` folder in VS Code.
2. Press `F5`.
3. VS Code uses `.vscode/launch.json` with `"type": "extensionHost"`.
4. A new Extension Development Host window opens.

## Manual Test Flow

In the Extension Development Host:

1. Open a Python workspace, such as the CodeScope repo or `examples/realistic_bugs/banking_app`.
2. Make sure CodeScope is installed in the Python environment used by the extension.
3. If needed, configure `codescope.pythonPath` to point at the correct Python executable.
4. Run `CodeScope: Index Workspace` from the command palette.
5. Run `CodeScope: Investigate Bug`.
6. Enter a bug description, for example:

```text
When I transfer money, the receiver balance does not increase
```

7. Open the `CodeScope Results` TreeView.
8. Confirm the tree shows:
   - Query
   - Likely relevant code
   - Related context
   - Result reasons as child items
9. Click a result such as `TransferService.transfer` and confirm the file opens at the reported start line.

## Settings

- `codescope.pythonPath`: Python executable used to run CodeScope. Default: `python`.
- `codescope.cliModule`: CodeScope CLI module. Default: `codescope.cli`.
- `codescope.maxResults`: Maximum investigate results requested from CodeScope. Default: `5`.

## Troubleshooting

- If `python` is not found, set `codescope.pythonPath`.
- If CodeScope is not installed, install it in the selected Python environment.
- If no index exists, run `CodeScope: Index Workspace`.
- If investigation fails, check the `CodeScope` output channel for stdout and stderr.
- Stderr may include third-party warnings; valid JSON stdout is what drives the result view.
