# Multi-Agent Planner

**LLM-powered system that solves coding tasks through agent collaboration.**

This project implements a multi-agent architecture using GPT-based agents, where each role has a specific function. A **Planner Agent** breaks down tasks, and a **Developer Agent** writes code to solve each subtask. The system is designed for extensibility and will evolve to include QA and Critic agents.

---

## Example Workflow

**Example user prompt:**  
*"Create a command-line tool that parses a CSV file and returns JSON-formatted summary statistics."*

System output:
- Planner identifies subtasks
- Developer writes code for each subtask
- QA Agent evaluates execution and correctness
- Critic Agent suggests refinements for failed code
- Developer revises and retries failed tasks
- Assembler generates a clean, deduplicated final program

---

## Folder Structure

```
multi_agent_planner/
├── agents/
│   ├── planner.py
│   ├── developer.py
│   ├── qa.py
│   ├── critic.py
│   └── base_agent.py
├── core/
│   ├── orchestrator.py
│   ├── memory.py
│   ├── task_schema.py
│   ├── assembler.py
│   └── assemble.py
├── prompts/
│   ├── planner_prompt.txt
│   └── developer_prompt.txt
├── output/
│   ├── session_log.json
│   ├── memory.json
│   └── final_program.py
├── main.py
├── requirements.txt
└── README.md
```

## Implemented Agents

- `PlannerAgent`: breaks task into subtasks
- `DeveloperAgent`: writes code for each subtask
- `QAAgent`: verifies execution and correctness of generated code
- `CriticAgent`: critiques and suggests code improvements
- `Assembler`: deduplicates and assembles final clean code output


## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI key:
   ```
   OPENAI_API_KEY=your-key-here
   ```

3. Run the planner:
   ```
   python main.py
   ```

# Note: Ensure your OpenAI API key has sufficient quota and access to GPT-4 or GPT-4o for best performance.

4. Follow the interactive prompt to generate your desired program.