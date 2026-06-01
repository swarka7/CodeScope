import * as path from "path";
import * as vscode from "vscode";

export function getWorkspaceRoot(): vscode.WorkspaceFolder | undefined {
  return vscode.workspace.workspaceFolders?.[0];
}

export function resolveWorkspaceFile(workspaceRoot: string, filePath: string): vscode.Uri {
  if (path.isAbsolute(filePath)) {
    return vscode.Uri.file(filePath);
  }

  return vscode.Uri.file(path.resolve(workspaceRoot, filePath));
}

export function toZeroBasedLine(lineNumber: number | undefined): number {
  if (lineNumber === undefined || !Number.isFinite(lineNumber)) {
    return 0;
  }

  return Math.max(0, Math.floor(lineNumber) - 1);
}
