# Multi-Agent Planner - Architecture Improvements

## Progress Tracker

| # | Improvement | Priority | Status | Notes |
|---|-------------|----------|--------|-------|
| 1 | Expand sandbox allowed imports | Quick Win | ✅ Done | Added uuid, dataclasses, typing, enum, pathlib, unittest, etc |
| 2 | Add retry loop with critic feedback | Medium | ✅ Done | develop_with_retry() in orchestrator, MAX_RETRIES=3 |
| 3 | Add shared context memory | Medium | ✅ Done | core/shared_context.py created, integrated into orchestrator |
| 4 | Smarter planner (fewer tasks) | Medium | ✅ Done | Better prompt, max 4 tasks, logical modules |
| 5 | Add Architect agent | Large | ✅ Done | agents/architect.py, creates design before coding |
| 6 | Add Integrator agent | Large | ✅ Done | agents/integrator.py, LLM-based intelligent merging |

---

## Problem Details

### 1. Expand Sandbox Allowed Imports
**Current**: `math, random, datetime, json, re, collections, itertools, functools, string, csv, io, statistics`

**Add**: `uuid, dataclasses, typing, enum, pathlib, abc, copy, operator`

**Why**: Tests fail because common Python stdlib modules are blocked.

**File**: `core/sandbox.py`

---

### 2. Add Retry Loop with Critic Feedback
**Current**: Critic reviews failed code but feedback is NOT sent back to developer for retry.

**Fix**: When QA fails, send critic feedback to developer and retry up to 3 times.

**Files**: `core/orchestrator.py`, `agents/developer.py`

---

### 3. Add Shared Context Memory
**Current**: Each task generates standalone code with no awareness of other tasks.

**Fix**: Create SharedContext class that stores:
- Architecture specification
- Previously generated code
- Interface definitions

**Files**: `core/shared_context.py` (NEW), update all agents

---

### 4. Smarter Planner (Fewer Tasks)
**Current**: Planner creates 8-10 micro-tasks like "implement add method".

**Fix**: Create 2-4 logical modules that are complete, testable units.

**File**: `agents/planner.py`

---

### 5. Add Architect Agent
**Current**: No high-level design phase - code structure emerges randomly.

**Fix**: New agent that creates:
- File structure
- Class/function signatures
- Dependencies between components

**File**: `agents/architect.py` (NEW)

---

### 6. Add Integrator Agent
**Current**: Assembler just concatenates code blocks naively.

**Fix**: New agent that:
- Merges code intelligently
- Resolves conflicts
- Ensures interfaces match
- Validates final output

**File**: `agents/integrator.py` (NEW), replace `core/assembler.py`

---

## Implementation Log

### Started: [timestamp]

(Updates will be added here as each improvement is completed)
