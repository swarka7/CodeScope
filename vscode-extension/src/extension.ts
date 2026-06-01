import * as vscode from "vscode";
import { CodeScopeRunner, CodeScopeRunnerError } from "./codescopeRunner";
import { getWorkspaceRoot } from "./pathUtils";
import { CodeScopeCodeResult } from "./types";

let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("CodeScope");
  const runner = new CodeScopeRunner(outputChannel);

  context.subscriptions.push(outputChannel);
  context.subscriptions.push(
    vscode.commands.registerCommand("codescope.indexWorkspace", () => indexWorkspace(runner)),
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("codescope.investigateBug", () => investigateBug(runner)),
  );
}

export function deactivate(): void {
  return;
}

async function indexWorkspace(runner: CodeScopeRunner): Promise<void> {
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
        title: "CodeScope: indexing workspace",
        cancellable: false,
      },
      () => runner.indexWorkspace(workspaceRoot),
    );

    if (run.exitCode === 0) {
      vscode.window.showInformationMessage("CodeScope index completed.");
      return;
    }

    outputChannel.show(true);
    vscode.window.showErrorMessage(
      `CodeScope index failed with exit code ${run.exitCode ?? "<unknown>"}. See the CodeScope output channel.`,
    );
  } catch (error) {
    showRunnerError("CodeScope index failed", error);
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
      outputChannel.show(true);
      vscode.window.showErrorMessage(
        response.message ?? "CodeScope investigate failed. See the CodeScope output channel.",
      );
      return;
    }

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
