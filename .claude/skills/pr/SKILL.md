---
name: pr
description: Create a pull request with automated linting, testing, and validation. Runs make lint, tests affected packages, collects PR details, and creates the PR.
tools: Bash, AskUserQuestion
---

# PR Creation Skill

Creates a pull request with automated linting, testing, and validation.

## When to Use

Invoke this skill when the user wants to create a pull request or says things like:
- "create a PR"
- "make a pull request"
- "push this and create PR"
- "/pr"

## Workflow

### Step 1: Check Current Branch

1. Check current branch with `git branch --show-current`
2. If on `main` or `master`, ask if they want to create a new feature branch
3. If yes, ask for branch name and create it with `git checkout -b <branch-name>`

### Step 2: Run Linting

**Important:** `make lint` runs on the entire repo and may report errors in unrelated files.
Always verify that staged/changed files pass lint independently.

1. Auto-fix lint and formatting on staged files:
   ```bash
   ruff check --fix .
   ruff format .
   ```
2. Verify all checks pass (this is what CI runs):
   ```bash
   ruff check .
   ruff format --check .
   ```
3. Common issues to watch for:
   - **F401** (unused imports): After refactoring, ensure old imports that are no longer referenced are removed
   - **I001** (unsorted imports): Adding a new symbol to an existing `from ... import` line may break alphabetical sorting — `ruff check --fix` auto-sorts these
4. If linting fails after auto-fix:
   - Show the errors with `ruff check .`
   - Use AskUserQuestion to ask: "Linting issues found. How to proceed?"
     - Options: "Fix manually and retry", "Show details", "Skip linting"
5. Stage any auto-fixed files before committing
6. Report linting status

### Step 3: Run Tests

1. Check `git status --short` to identify modified files
2. Parse modified files and determine affected packages:
   - Files in `util/opentelemetry-util-genai-evals-deepeval/` → test `util/opentelemetry-util-genai-evals-deepeval`
   - Files in `util/opentelemetry-util-genai-evals/` → test `util/opentelemetry-util-genai-evals`
   - Files in `util/opentelemetry-util-genai-emitters-splunk/` → test `util/opentelemetry-util-genai-emitters-splunk`
   - Files in `util/opentelemetry-util-genai/` → test `util/opentelemetry-util-genai`
   - Files in `instrumentation-genai/<package>/` → test that specific package
3. For each affected package:
   - Run `pytest <package>/tests/ -v`
   - If tests fail, use AskUserQuestion:
     - Options: "Show failures and help fix", "Skip tests", "Abort PR"
4. Report test results

### Step 4: Collect PR Information

1. Check if ticket number was provided as skill argument
2. If not, use AskUserQuestion to ask for ticket number (optional)
3. Get recent commits: `git log origin/main..HEAD --pretty=format:"- %s"`
4. Get most recent commit for title suggestion: `git log -1 --pretty=%B`
5. Use AskUserQuestion to confirm or modify PR title
6. Generate PR description template:

```markdown
## Summary
<commit list from git log>

## Test plan
- [x] Linting passed
- [x] Tests passed for <affected packages>

## Related Issues
- <ticket-number if provided>

🤖 Generated with [Claude Code](https://claude.ai/code)
```

7. Use AskUserQuestion to ask if they want to use the generated description or write their own

### Step 5: Commit Linting Fixes

1. Check for uncommitted changes: `git status --short`
2. If changes exist, commit them:

```bash
git add .
git commit -m "chore: apply linting fixes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Step 6: Push and Create PR

1. Get current branch: `git branch --show-current`
2. Push to remote: `git push -u origin <branch-name>`
3. Create PR with gh CLI:

```bash
gh pr create --title "<title>" --body "<description>"
```

4. Display PR URL to user

## Error Handling

- **gh CLI not installed**: Show error and provide installation command for macOS: `brew install gh`
- **Not authenticated**: Run `gh auth login --git-protocol ssh --web` with timeout
- **Linting failures**: Show errors, ask user for guidance
- **Test failures**: Show details, offer to help debug
- **Push fails**: Show error, check if remote exists

## Example Invocation

User: `/pr HYBIM-492`

Agent:
1. Checks branch (not on main)
2. Runs `make lint` → ✓ Passed
3. Identifies affected packages from `git status`
4. Runs tests for each package → ✓ All passed
5. Asks about PR title (suggests from recent commit)
6. Generates and confirms PR description
7. Pushes branch and creates PR
8. Shows PR URL

## Notes

- Use conventional commit prefixes in titles: feat:, fix:, chore:, docs:, test:, refactor:
- Always verify not on main branch before pushing
- Include ticket number in title if provided
- Run all validation before creating PR
- Use AskUserQuestion for all user input to maintain flow
