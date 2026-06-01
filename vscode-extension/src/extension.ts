import * as vscode from "vscode";
import { CodeScopeRunner, CodeScopeRunnerError } from "./codescopeRunner";
import { getWorkspaceRoot } from "./pathUtils";
import { CodeScopeResultsProvider, openResult } from "./resultView";
import { CodeScopeCodeResult } from "./types";

let outputChannel: vscode.OutputChannel;
let resultsProvider: CodeScopeResultsProvider;

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("CodeScope");
  resultsProvider = new CodeScopeResultsProvider();
  const runner = new CodeScopeRunner(outputChannel);

  context.subscriptions.push(outputChannel);
  context.subscriptions.push(
    vscode.window.createTreeView("codescope.resultsView", {
      treeDataProvider: resultsProvider,
      showCollapseAll: true,
    }),
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("codescope.indexWorkspace", () => indexWorkspace(runner)),
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("codescope.rebuildIndex", () => rebuildIndex(runner)),
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("codescope.investigateBug", () => investigateBug(runner)),
  );
  context.subscriptions.push(vscode.commands.registerCommand("codescope.openResult", openResult));
}

export function deactivate(): void {
  return;
}

async function indexWorkspace(runner: CodeScopeRunner): Promise<void> {
  await runIndexCommand(runner, {
    progressTitle: "CodeScope: indexing workspace",
    successMessage: "CodeScope index completed.",
    failurePrefix: "CodeScope index failed",
    rebuild: false,
  });
}

async function rebuildIndex(runner: CodeScopeRunner): Promise<void> {
  await runIndexCommand(runner, {
    progressTitle: "CodeScope: rebuilding index",
    successMessage: "CodeScope index rebuild completed.",
    failurePrefix: "CodeScope index rebuild failed",
    rebuild: true,
  });
}

async function runIndexCommand(
  runner: CodeScopeRunner,
  options: {
    progressTitle: string;
    successMessage: string;
    failurePrefix: string;
    rebuild: boolean;
  },
): Promise<void> {
  const workspace = getWorkspaceRoot();
  if (!workspace) {
    vscode.window.showErrorMessage("Open a workspace before indexing with CodeScope.");
    return;
  }

  try {
    const workspaceRoot = workspace.uri.fsPath;
    const run = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: options.progressTitle,
        cancellable: false,
      },
      () =>
        options.rebuild
          ? runner.rebuildWorkspace(workspaceRoot)
          : runner.indexWorkspace(workspaceRoot),
    );

    if (run.exitCode === 0) {
      vscode.window.showInformationMessage(options.successMessage);
      return;
    }

    outputChannel.show(true);
    vscode.window.showErrorMessage(
      `${options.failurePrefix} with exit code ${run.exitCode ?? "<unknown>"}. See the CodeScope output channel.`,
    );
  } catch (error) {
    showRunnerError(options.failurePrefix, error);
  }
}

async function investigateBug(runner: CodeScopeRunner): Promise<void> {
  const workspace = getWorkspaceRoot();
  if (!workspace) {
    vscode.window.showErrorMessage("Open a workspace before running CodeScope investigate.");
    return;
  }

  const description = await vscode.window.showInputBox({
    title: "CodeScope: Investigate Bug",
    prompt: "Describe the bug",
    placeHolder: "When I transfer money, the receiver balance does not increase",
    ignoreFocusOut: true,
  });

  if (description === undefined) {
    return;
  }

  const trimmedDescription = description.trim();
  if (!trimmedDescription) {
    vscode.window.showWarningMessage("Enter a bug description before running CodeScope investigate.");
    return;
  }

  try {
    const workspaceRoot = workspace.uri.fsPath;
    const investigation = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "CodeScope: investigating bug",
        cancellable: false,
      },
      () => runner.investigate(workspaceRoot, trimmedDescription),
    );

    const response = investigation.response;
    if (response.status === "error") {
      const message = response.message ?? "CodeScope investigate returned an error.";
      resultsProvider.setError(message);
      vscode.commands.executeCommand("codescope.resultsView.focus");
      outputChannel.show(true);
      const action = await vscode.window.showErrorMessage(message, "Rebuild Index");
      if (action === "Rebuild Index") {
        await vscode.commands.executeCommand("codescope.rebuildIndex");
      }
      return;
    }

    resultsProvider.setInvestigationResult(response, workspaceRoot);
    vscode.commands.executeCommand("codescope.resultsView.focus");

    const topResult = response.likely_relevant_code[0];
    if (!topResult) {
      vscode.window.showInformationMessage("CodeScope investigation completed with no results.");
      return;
    }

    vscode.window.showInformationMessage(formatTopResultMessage(topResult));
  } catch (error) {
    showRunnerError("CodeScope investigate failed", error);
  }
}

function formatTopResultMessage(result: CodeScopeCodeResult): string {
  return `CodeScope top result: ${result.name} (${result.file_path}:${result.start_line}-${result.end_line})`;
}

function showRunnerError(prefix: string, error: unknown): void {
  outputChannel.show(true);
  if (error instanceof CodeScopeRunnerError) {
    vscode.window.showErrorMessage(error.message);
    return;
  }

  vscode.window.showErrorMessage(`${prefix}: ${String(error)}`);
}
