# Skill Plugins

## Architecture

Skills are the modular "capabilities" of the system. Each skill is a self-contained directory under `.claude/skills/` with:

```
.claude/skills/
├── ask-question/          # RAG knowledge base Q&A
│   ├── SKILL.md           # Metadata (YAML frontmatter)
│   ├── script/
│   │   └── agent.py       # Agent implementation
│   └── data/              # Knowledge base files
├── plan-trip/             # Itinerary planning
├── query-info/            # Web information search
├── preference/            # User preference management
├── memory-query/          # Memory/history queries
├── rag-experience/        # RAG experience tips extraction
├── rag-risk/              # RAG risk/pitfall extraction
├── event-collection/      # Local event collection
├── accommodation-query/   # Hotel/accommodation search
└── README.md
```

## SKILL.md Format

Each skill has a `SKILL.md` file with YAML frontmatter:

```yaml
---
name: plan-trip
description: >
  Generate a multi-day travel itinerary based on origin, destination,
  dates, and user preferences. Outputs structured daily plans.
---

## Detailed Instructions

(Markdown body used as execution-time prompt injection)
```

The `SkillLoader` class (`utils/skill_loader.py`) parses these files to:
1. Build the skill catalog for intent recognition prompts
2. Inject skill-specific instructions into agent execution prompts

## LazyAgentRegistry

Defined in `agents/lazy_agent_registry.py`, the registry provides lazy-loading and caching of skill agents.

### How It Works

1. **Discovery**: On init, scans `.claude/skills/*/script/agent.py` to build a skill map
2. **Resolution**: Maps agent names to skill directories (supports legacy name aliases)
3. **Lazy Loading**: On first access (`registry["agent_name"]`), dynamically imports the module, finds the Agent class (any class with a `run` method), and instantiates it
4. **Caching**: Subsequent accesses return the cached instance
5. **Dependency Injection**: Automatically injects `model` and `memory_manager` (if the agent's `__init__` accepts it)

### Legacy Name Mapping

For backward compatibility, the registry maps old agent names to skill directories:

| Agent Name | Skill Directory |
|------------|----------------|
| `rag_knowledge` | `ask-question` |
| `rag_experience` | `rag-experience` |
| `rag_risk` | `rag-risk` |
| `memory_query` | `memory-query` |
| `preference` | `preference` |
| `information_query` | `query-info` |
| `itinerary_planning` | `plan-trip` |
| `event_collection` | `event-collection` |

### Standalone Agents

Two agents are registered directly in `build_graph()` rather than loaded from skills:

- **`TransportAgent`** (`agents/transport_agent.py`): Queries flight and train MCP services
- **`POIFetchAgent`** (`agents/poi_agent.py`): Fetches POI candidates via Amap search

## Writing a New Skill

1. Create a directory under `.claude/skills/`:
   ```
   .claude/skills/my-skill/
   ├── SKILL.md
   └── script/
       └── agent.py
   ```

2. Write `SKILL.md` with frontmatter:
   ```yaml
   ---
   name: my-skill
   description: A brief description of what this skill does.
   ---
   ```

3. Implement `agent.py` with a class that has a `run` method:
   ```python
   class MySkillAgent:
       def __init__(self, name: str, model, memory_manager=None):
           self.name = name
           self.model = model
           self.memory_manager = memory_manager

       async def run(self, context: dict) -> dict:
           # context contains user_query, hard_constraints, etc.
           result = ...
           return {
               "agent_name": self.name,
               "status": "success",
               "data": result,
           }
   ```

4. Add a legacy mapping in `LazyAgentRegistry.__init__` if the intent node uses a different name:
   ```python
   self._legacy_mapping["my_intent_name"] = "my-skill"
   ```

5. The skill will be auto-discovered on next startup.
