# Kuiskaus — Agent Guidelines

## Context7 Protocol

Before writing ANY code, check Context7 for current documentation on:
- `mlx-whisper` — MLX Whisper API and model loading patterns
- `rumps` — macOS menu bar app framework
- `pyobjc` — Python/Objective-C bridge for macOS APIs
- `pyaudio` — audio recording API

Training data may be outdated. Context7 provides authoritative, up-to-date docs.

---

## Minimalist Engineering Philosophy

**Every line of code is a liability.** Before creating anything:

- **LESS IS MORE**: Question necessity before creation
- **Challenge Everything**: Ask "Is this truly needed?" before implementing
- **Minimal Viable Solution**: Build the simplest thing that fully solves the problem
- **No Speculative Features**: Don't build for "future needs" - solve today's problem
- **Prefer Existing**: Reuse existing code/tools before creating new ones
- **One Purpose Per Component**: Each function/module should do one thing well

### Pre-Creation Challenge (MANDATORY)

Before creating ANY code, ask:
1. Is this explicitly required by the GitHub issue?
2. Can existing code/tools solve this instead?
3. What's the SIMPLEST way to meet the requirement?
4. Will removing this break core functionality?
5. Am I building for hypothetical future needs?

**If you cannot justify the necessity, DO NOT CREATE IT.**

---

## Pre-Push Quality Gates

All of the following MUST pass locally before any `git push`. CI is for verification, not discovery.

```bash
# Run full test suite
./run_tests.sh

# Linting
uv run ruff check kuiskaus/ tests/

# Type checking
uv run ty check kuiskaus/

# Format check
uv run ruff format --check kuiskaus/ tests/
```

**If any check fails**: Fix the issue. NEVER bypass with blanket suppressions. The ONE accepted suppression is a per-line `# noqa: <CODE> - <reason>` on a broad `except Exception` guard that logs the exception, at a boundary where catching a specific exception set would be arbitrary (worker threads, CGEvent/NSEvent run-loop callbacks, the quit path, test runners). Everything else remains prohibited: global ignore lists, file-level suppressions, blanket `# noqa` without a code, and `# type: ignore`.

---

## Testing Standards

- **TDD preferred**: Write tests before implementation
- **Coverage threshold**: 80%+ for all new code
- **Test location**: `tests/` directory, mirroring `kuiskaus/` module structure
- **Test runner**: `./run_tests.sh` (runs pytest; add `--hardware` to also run the manual hardware scripts)

### macOS Hardware Dependencies

Many components (audio, hotkeys, text insertion) require real macOS hardware. For these:
- Use mocks/stubs in unit tests for hardware-dependent code
- Integration tests (`test_integration.py`) may require a running macOS environment
- Mark hardware-dependent tests with `@pytest.mark.skipif` when running in CI

### Coverage

```bash
uv run pytest --cov=kuiskaus --cov-report=term-missing tests/
```

---

## Code Style & Conventions

- **Python version**: 3.11+ (authoritative source: `requires-python` in `pyproject.toml`)
- **Formatter**: ruff format
- **Linter**: ruff check
- **Type checker**: ty
- **Line length**: 88 characters (ruff default)
- **Imports**: isort-compatible (ruff handles this)

### Apple Silicon Only

This project is **exclusively for Apple Silicon Macs**. Do NOT add Intel/x86 compatibility code. If a feature requires hardware, assert it clearly and fail fast.

### Module Size

- **Hard limit**: 500 lines per file
- **Ideal**: 300 lines or fewer
- **Refactor trigger**: File exceeds 500 lines or has 3+ distinct responsibilities

---

## Git Workflow

### Commit Format (Conventional Commits)

```
type(scope): description

feat(hotkey): add configurable modifier key support
fix(transcriber): handle empty audio buffer gracefully
refactor(menubar): extract status display into helper
test(audio): add unit tests for recorder edge cases
docs(readme): update installation instructions
```

**Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

### Branch Naming

```
feature/issue-{number}-short-description
fix/issue-{number}-short-description
```

### PR Process

1. All quality gates pass locally
2. PR description includes `Fixes #<issue-number>`
3. Tests added for new functionality
4. No bypasses (`# noqa`, `# type: ignore`, etc.)

---

## Documentation Policy

### The 200-PR Test

Before creating documentation, ask: "Will this be true in 200 PRs?"

- **YES** → Document the principle (WHY)
- **NO** → Skip it or use inline code comments

### Forbidden Temporary Files

Never commit:
- `RESEARCH.md`, `ANALYSIS.md`, `IMPLEMENTATION_PLAN.md`
- `TODO.md`, `NOTES.md`, `SCRATCH.md`
- Any ALL_CAPS `.md` files that describe transient work

If work must be deferred, create a GitHub issue. **The issue IS the TODO.**

---

## Commands Reference

```bash
# Setup
./setup.sh                          # Full install (uv, deps, model download)

# Run application
./launch_kuiskaus.sh                # Menu bar app
./launch_cli.sh                     # CLI version

# Tests
./run_tests.sh                      # Unit test suite (pytest)
./run_tests.sh --hardware           # Also run manual hardware scripts
uv run pytest tests/test_transcriber.py   # Single test file
uv run pytest -k "test_name"        # Single test by name
uv run pytest --cov=kuiskaus tests/ # With coverage

# Linting & formatting
uv run ruff check kuiskaus/ tests/           # Lint
uv run ruff check --fix kuiskaus/ tests/     # Lint + auto-fix
uv run ruff format kuiskaus/ tests/          # Format
uv run ruff format --check kuiskaus/ tests/  # Format check only

# Type checking
uv run ty check kuiskaus/

# Dependency management
uv sync                # Install runtime deps
uv sync --group dev    # Install all deps including dev tools (ty)
uv lock                # Regenerate uv.lock after editing pyproject.toml
```

---

## Merge Authority

Agents MAY merge pull requests, but ONLY when every condition below holds:

1. **All quality gates pass locally, with no exceptions** — `./run_tests.sh`, `uv run ruff check kuiskaus/ tests/`, `uv run ty check kuiskaus/`, `uv run ruff format --check kuiskaus/ tests/`. There is no exemption mechanism and no such thing as a "known-failing" gate: a red gate blocks the merge, full stop.
2. **CI passes**, where CI exists.
3. **Any adversarial or code-review gate attached to the work returned an approving verdict.**
4. **The diff stays within the scope its issue declares.** A PR touching files outside the issue's stated scope or diff budget must be halted and reported — never merged, and never merged "because the extra changes look harmless".

**Pre-existing failures are not grounds for an exemption.** If a gate fails because of issues that pre-date the PR's diff, those issues must be fixed — on this PR or on an unblocking branch — before anything merges. A PR may not land while any quality gate is red, regardless of when the failure originated.

### When any condition is unmet

Stop at PR creation, apply the `needs-human-attention` label, and notify the operator with the specific failing condition. **When in doubt, do not merge.**