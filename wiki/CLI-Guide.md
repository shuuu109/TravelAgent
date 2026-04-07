# CLI Guide

The interactive CLI is the primary user interface, built with the [Rich](https://github.com/Textualize/rich) library for terminal rendering.

## Starting the CLI

```bash
python cli.py
```

On startup, you'll be prompted for a user ID (default: `default_user`). The system then initializes the LangGraph workflow, memory manager, and circuit breaker.

## Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands and example queries |
| `status` | Display current session state, constraints, memory stats, and circuit breaker status |
| `health` | Run a health check against the LLM service |
| `clear` | Reset current task state (preserves long-term memory) |
| `history` | View past trip records from long-term memory |
| `preferences` | Display saved user preferences |
| `exit` | End session and exit |

Any input that doesn't match a command is treated as a **natural language query** and sent through the LangGraph pipeline.

## Example Queries

```
我要从上海去北京旅游3天
# → Triggers itinerary planning with POI search, routing, hotels

北京有哪些必去景点
# → Information query via web search

我喜欢住万豪酒店
# → Preference extraction and persistence

查一下明天北京的天气
# → Information query

帮我回忆上次去杭州的行程
# → Memory query from long-term storage
```

## Query Processing Flow

When you enter a natural language query:

1. **Circuit breaker check** — if the service is in OPEN state, the query is rejected
2. **Long-term memory loading** — retrieves user preferences, past trip summaries, and relevant history
3. **Short-term context** — gathers the last 5 turns of dialogue
4. **LangGraph invocation** — the full pipeline runs (intent → validate → orchestrate → plan → respond)
5. **Result display**:
   - Shows which agents were called and their status (✓/✗)
   - Renders the final response as Markdown
6. **Memory update** — adds both user input and assistant response to short-term and long-term memory

## Status Command Output

The `status` command displays:

- **Session info**: User ID, session ID
- **Hard constraints**: Origin, destination, dates, completeness
- **Memory stats**: Short-term message count, long-term totals
- **Circuit breaker**: Current state, failure count
- **Loaded agents**: Which skill agents have been instantiated

## Display Features

The CLI uses Rich components for formatting:

- **`Panel`**: Framed sections for structured output
- **`Table`**: Tabular data (commands, preferences, history)
- **`Markdown`**: Final response rendering with headers, lists, and emphasis
- **`Progress`**: Spinner during processing ("思考中...")
- **Color coding**: Green for success, red for errors, yellow for warnings, cyan for highlights
