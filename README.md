

# Multi-Agent Planner

**LLM-powered system that solves coding tasks through agent collaboration.**

This project implements a multi-agent architecture using GPT-based agents, where each role has a specific function. A **Planner Agent** breaks down tasks, and a **Developer Agent** writes code to solve each subtask. The system is designed for extensibility and will evolve to include QA and Critic agents.

---

## Example Workflow

User prompt:
> "Build a CLI tool that reads a CSV file and outputs JSON summary stats."

System output:
- Planner identifies subtasks:
  1. Parse CLI arguments
  2. Read CSV
  3. Calculate summary stats
  4. Write JSON output
- Developer writes code for each subtask
- (Future: QA agent tests and critiques results)

---

## Folder Structure

```
multi_agent_planner/
├── agents/
│   ├── planner.py              # PlannerAgent: breaks tasks into subtasks
│   ├── developer.py            # DeveloperAgent: generates code for each task
│   └── base_agent.py           # Optional: shared methods/utilities for agents
├── core/
│   ├── orchestrator.py         # Main loop coordinator (for future)
│   ├── memory.py               # Task/result memory module (for future)
│   └── task_schema.py          # JSON schema for structured tasks (optional)
├── prompts/
│   ├── planner_prompt.txt      # Prompt template for planner role
│   └── developer_prompt.txt    # Prompt template for developer role
├── examples/
│   └── test_task_1.json        # Saved example outputs or inputs
├── main.py                     # Project entrypoint
├── requirements.txt            # Dependencies list
└── README.md                   # Project description and usage
```

## Planned Agents

- `PlannerAgent`: breaks task into subtasks
- `DeveloperAgent`: writes code for each subtask
- `QAAgent` (coming soon): evaluates code results
- `CriticAgent` (optional): identifies flaws and improvements
- `Coordinator`: manages agent sequence and feedback loops