export type CodeScopeStatus = "ok" | "error";

export interface CodeScopeCodeResult {
  rank: number;
  name: string;
  kind: string;
  file_path: string;
  start_line: number;
  end_line: number;
  source: string;
  score: number | null;
  reasons: string[];
  dependencies: string[];
}

export interface CodeScopeInvestigationResponse {
  schema_version: number;
  status: CodeScopeStatus;
  repo: string;
  query: string;
  message?: string;
  likely_relevant_code: CodeScopeCodeResult[];
  related_context: CodeScopeCodeResult[];
}

export interface CodeScopeRunResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

export interface CodeScopeInvestigationRun {
  response: CodeScopeInvestigationResponse;
  run: CodeScopeRunResult;
}
