# CodeScope VS Code Extension

Minimal VS Code extension prototype for running CodeScope from inside VS Code.

This v0 scaffold supports:

- `CodeScope: Index Workspace`
- `CodeScope: Investigate Bug`
- Running `python -m codescope.cli investigate <workspace> "<description>" --json`
- Logging full stdout and stderr to a `CodeScope` output channel
- Showing a simple summary of the top investigation result

TreeView results and clickable navigation are planned for the next step.

## Development

Install extension dependencies:

```bash
npm install
```

Compile TypeScript:

```bash
npm run compile
```

Launch the extension in VS Code:

1. Open this repository in VS Code.
2. Open the `vscode-extension/` folder or use it as the extension project.
3. Press `F5` to start an Extension Development Host.
4. Open a Python workspace that has CodeScope installed and indexed.
5. Run `CodeScope: Investigate Bug` from the command palette.

## Settings

- `codescope.pythonPath`: Python executable used to run CodeScope. Default: `python`.
- `codescope.cliModule`: CodeScope CLI module. Default: `codescope.cli`.
- `codescope.maxResults`: Maximum investigate results requested from CodeScope. Default: `5`.

## Notes

This extension calls the local CodeScope CLI. It does not call `--llm`, does not generate patches, and does not modify user files.
