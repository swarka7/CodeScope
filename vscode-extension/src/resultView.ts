import * as vscode from "vscode";
import { resolveWorkspaceFile, toZeroBasedLine } from "./pathUtils";
import { CodeScopeCodeResult, CodeScopeInvestigationResponse } from "./types";

type ResultTreeNode =
  | EmptyNode
  | ErrorNode
  | QueryNode
  | SectionNode
  | ResultNode
  | ReasonNode;

interface EmptyNode {
  type: "empty";
  label: string;
}

interface ErrorNode {
  type: "error";
  label: string;
}

interface QueryNode {
  type: "query";
  query: string;
}

interface SectionNode {
  type: "section";
  label: string;
  children: ResultNode[] | EmptyNode[];
}

interface ResultNode {
  type: "result";
  result: CodeScopeCodeResult;
  workspaceRoot: string;
  children: ReasonNode[];
}

interface ReasonNode {
  type: "reason";
  reason: string;
}

export class CodeScopeResultsProvider implements vscode.TreeDataProvider<ResultTreeNode> {
  private readonly onDidChangeTreeDataEmitter = new vscode.EventEmitter<
    ResultTreeNode | undefined | null | void
  >();

  readonly onDidChangeTreeData = this.onDidChangeTreeDataEmitter.event;
  private roots: ResultTreeNode[] = [{ type: "empty", label: "No results yet." }];

  setInvestigationResult(response: CodeScopeInvestigationResponse, workspaceRoot: string): void {
    if (response.status === "error") {
      this.setError(response.message ?? "CodeScope investigate returned an error.");
      return;
    }

    this.roots = [
      { type: "query", query: response.query },
      this.createSection("Likely relevant code", response.likely_relevant_code, workspaceRoot),
      this.createSection("Related context", response.related_context, workspaceRoot),
    ];
    this.onDidChangeTreeDataEmitter.fire();
  }

  setError(message: string): void {
    this.roots = [{ type: "error", label: message }];
    this.onDidChangeTreeDataEmitter.fire();
  }

  getTreeItem(element: ResultTreeNode): vscode.TreeItem {
    switch (element.type) {
      case "empty":
        return new vscode.TreeItem(element.label, vscode.TreeItemCollapsibleState.None);
      case "error":
        return this.createErrorItem(element);
      case "query":
        return this.createQueryItem(element);
      case "section":
        return this.createSectionItem(element);
      case "result":
        return this.createResultItem(element);
      case "reason":
        return this.createReasonItem(element);
    }
  }

  getChildren(element?: ResultTreeNode): ResultTreeNode[] {
    if (!element) {
      return this.roots;
    }

    if (element.type === "section" || element.type === "result") {
      return element.children;
    }

    return [];
  }

  private createSection(
    label: string,
    results: CodeScopeCodeResult[],
    workspaceRoot: string,
  ): SectionNode {
    if (results.length === 0) {
      return {
        type: "section",
        label,
        children: [{ type: "empty", label: `No ${label.toLowerCase()} returned.` }],
      };
    }

    return {
      type: "section",
      label,
      children: results.map((result) => ({
        type: "result",
        result,
        workspaceRoot,
        children: this.createReasonNodes(result),
      })),
    };
  }

  private createReasonNodes(result: CodeScopeCodeResult): ReasonNode[] {
    if (result.reasons.length === 0) {
      return [{ type: "reason", reason: "No reasons returned." }];
    }

    return result.reasons.map((reason) => ({ type: "reason", reason }));
  }

  private createErrorItem(element: ErrorNode): vscode.TreeItem {
    const item = new vscode.TreeItem(element.label, vscode.TreeItemCollapsibleState.None);
    item.iconPath = new vscode.ThemeIcon("error");
    return item;
  }

  private createQueryItem(element: QueryNode): vscode.TreeItem {
    const item = new vscode.TreeItem("Query", vscode.TreeItemCollapsibleState.None);
    item.description = element.query;
    item.tooltip = element.query;
    item.iconPath = new vscode.ThemeIcon("search");
    return item;
  }

  private createSectionItem(element: SectionNode): vscode.TreeItem {
    const item = new vscode.TreeItem(element.label, vscode.TreeItemCollapsibleState.Expanded);
    item.iconPath = new vscode.ThemeIcon("list-tree");
    return item;
  }

  private createResultItem(element: ResultNode): vscode.TreeItem {
    const result = element.result;
    const item = new vscode.TreeItem(
      `${result.rank}. ${result.name}`,
      vscode.TreeItemCollapsibleState.Collapsed,
    );
    item.description = `${result.file_path}:${result.start_line}-${result.end_line}`;
    item.tooltip = createResultTooltip(result);
    item.iconPath = new vscode.ThemeIcon(result.source === "related" ? "references" : "symbol-method");
    item.command = {
      command: "codescope.openResult",
      title: "Open CodeScope Result",
      arguments: [element],
    };
    return item;
  }

  private createReasonItem(element: ReasonNode): vscode.TreeItem {
    const item = new vscode.TreeItem(element.reason, vscode.TreeItemCollapsibleState.None);
    item.iconPath = new vscode.ThemeIcon("debug-breakpoint-log");
    return item;
  }
}

export async function openResult(node: ResultNode): Promise<void> {
  const result = node.result;
  const uri = resolveWorkspaceFile(node.workspaceRoot, result.file_path);
  const startLine = toZeroBasedLine(result.start_line);
  const endLine = Math.max(startLine, toZeroBasedLine(result.end_line));
  await vscode.window.showTextDocument(uri, {
    selection: new vscode.Range(startLine, 0, endLine, 0),
  });
}

function createResultTooltip(result: CodeScopeCodeResult): string {
  const score = result.score === null ? "<none>" : result.score.toFixed(2);
  const reasons = result.reasons.length > 0 ? result.reasons.join(", ") : "<none>";
  return [
    `${result.name}`,
    `Kind: ${result.kind}`,
    `Source: ${result.source}`,
    `Score: ${score}`,
    `Location: ${result.file_path}:${result.start_line}-${result.end_line}`,
    `Reasons: ${reasons}`,
  ].join("\n");
}
