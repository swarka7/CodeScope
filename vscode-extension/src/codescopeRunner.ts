import { spawn } from "child_process";
import * as vscode from "vscode";
import {
  CodeScopeInvestigationResponse,
  CodeScopeInvestigationRun,
  CodeScopeRunResult,
} from "./types";

export class CodeScopeRunnerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CodeScopeRunnerError";
  }
}

export class CodeScopeRunner {
  constructor(private readonly outputChannel: vscode.OutputChannel) {}

  async indexWorkspace(workspaceRoot: string): Promise<CodeScopeRunResult> {
    return this.run(["index", workspaceRoot], workspaceRoot);
  }

  async rebuildWorkspace(workspaceRoot: string): Promise<CodeScopeRunResult> {
    return this.run(["index", workspaceRoot, "--rebuild"], workspaceRoot);
  }

  async investigate(workspaceRoot: string, description: string): Promise<CodeScopeInvestigationRun> {
    const maxResults = getMaxResults();
    const run = await this.run(
      ["investigate", workspaceRoot, description, "--top-k", String(maxResults), "--json"],
      workspaceRoot,
    );

    let response: CodeScopeInvestigationResponse;
    try {
      response = JSON.parse(run.stdout) as CodeScopeInvestigationResponse;
    } catch (error) {
      throw new CodeScopeRunnerError(
        `CodeScope returned output that was not valid JSON. See the CodeScope output channel for details.`,
      );
    }

    return { response, run };
  }

  private run(args: string[], cwd: string): Promise<CodeScopeRunResult> {
    const pythonPath = getConfigString("pythonPath", "python");
    const cliModule = getConfigString("cliModule", "codescope.cli");
    const commandArgs = ["-m", cliModule, ...args];

    this.logCommand(pythonPath, commandArgs, cwd);

    return new Promise((resolve, reject) => {
      const child = spawn(pythonPath, commandArgs, {
        cwd,
        shell: false,
        windowsHide: true,
      });

      let stdout = "";
      let stderr = "";

      child.stdout.setEncoding("utf8");
      child.stderr.setEncoding("utf8");

      child.stdout.on("data", (chunk: string) => {
        stdout += chunk;
      });

      child.stderr.on("data", (chunk: string) => {
        stderr += chunk;
      });

      child.on("error", (error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") {
          reject(
            new CodeScopeRunnerError(
              `Python executable not found: ${pythonPath}. Configure codescope.pythonPath.`,
            ),
          );
          return;
        }

        reject(new CodeScopeRunnerError(`Failed to run CodeScope: ${error.message}`));
      });

      child.on("close", (exitCode) => {
        this.logResult(stdout, stderr, exitCode);
        resolve({ exitCode, stdout, stderr });
      });
    });
  }

  private logCommand(pythonPath: string, args: string[], cwd: string): void {
    this.outputChannel.appendLine("");
    this.outputChannel.appendLine("CodeScope command");
    this.outputChannel.appendLine(`cwd: ${cwd}`);
    this.outputChannel.appendLine(`command: ${pythonPath} ${args.join(" ")}`);
  }

  private logResult(stdout: string, stderr: string, exitCode: number | null): void {
    this.outputChannel.appendLine(`exit code: ${exitCode ?? "<unknown>"}`);
    this.outputChannel.appendLine("");
    this.outputChannel.appendLine("stdout:");
    this.outputChannel.appendLine(stdout.trim() || "<empty>");
    this.outputChannel.appendLine("");
    this.outputChannel.appendLine("stderr:");
    this.outputChannel.appendLine(stderr.trim() || "<empty>");
  }
}

function getConfigString(key: string, fallback: string): string {
  const value = vscode.workspace.getConfiguration("codescope").get<string>(key);
  if (!value || !value.trim()) {
    return fallback;
  }

  return value.trim();
}

function getMaxResults(): number {
  const value = vscode.workspace.getConfiguration("codescope").get<number>("maxResults", 5);
  if (!Number.isFinite(value)) {
    return 5;
  }

  return Math.max(1, Math.min(20, Math.floor(value)));
}
