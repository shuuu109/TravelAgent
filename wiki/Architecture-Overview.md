# Architecture Overview

## System Overview

Aligo uses a **Plan-and-Execute** architecture built on LangGraph. User input flows through a pipeline of graph nodes that identify intent, orchestrate skill agents in parallel, plan itineraries, and generate natural language responses.

```
User Input
   |
   v
+-------------------------------------------------------------+
|  P1: intent_node (Intent Recognition)                       |
|  - Semantic intent classification (6 categories)            |
|  - Key entity extraction (origin, destination, dates)       |
|  - Agent scheduling plan with priorities                    |
+-------------------------------------------------------------+
   |
   v
+-------------------------------------------------------------+
|  P1.5: validate_constraints (Rule Guard)                    |
|  - Amap MCP-based distance/time feasibility check           |
|  - Routes to negotiate_constraints on violations            |
+-------------------------------------------------------------+
   |
   v
+-------------------------------------------------------------+
|  P2: orchestrate_node (Parallel Dispatch)                   |
|  - Groups agents by priority                                |
|  - Same-priority agents run via asyncio.gather              |
|  - Enriches context between priority batches                |
+-------------------------------------------------------------+
   |
   v
+-------------------------------------------------------------+
|  P3: itinerary_planning_node                                |
|  - POI selection, daily clustering, route optimization      |
|  - TSP-based route ordering with Amap distance matrix       |
|  - Restaurant recommendations via Amap nearby search        |
+-------------------------------------------------------------+
   |
   v
+-------------------------------------------------------------+
|  P3.5: poi_enrich_node (Post-Retrieval Augmentation)        |
|  - RAG-based experience descriptions for each POI           |
|  - Writes poi_descriptions to state                         |
+-------------------------------------------------------------+
   |
   v
+-------------------------------------------------------------+
|  P4: accommodation_node                                     |
|  - Hotel recommendations via RollingGo MCP                  |
|  - Considers arrival hub and user preferences               |
+-------------------------------------------------------------+
   |
   v
+-------------------------------------------------------------+
|  P4.5: itinerary_review_node (Self-Check)                   |
|  - Validates itinerary against physical constraints          |
|  - Can loop back to P3 once for re-planning                 |
+-------------------------------------------------------------+
   |
   v
+-------------------------------------------------------------+
|  P5: respond_node (Response Generation)                     |
|  - Structured daily route rendering                         |
|  - RAG tips and risk warnings                               |
|  - LLM fallback summarization                               |
+-------------------------------------------------------------+
   |
   v
Final Response to User
```

## LangGraph Workflow

The graph is built in `graph/workflow.py` via `build_graph()`. It compiles a `StateGraph[TravelGraphState]` with conditional edges:

```
START -> intent -> validate_constraints
                      |
            +---------+---------+
            |                   |
        orchestrate          negotiate -> END
            |
    itinerary_planning
            |
        poi_enrich
            |
      accommodation
            |
    itinerary_review
            |
      +-----+-----+
      |           |
   respond    itinerary_planning  (retry loop, max 1)
      |
     END
```

### Conditional Routing

1. **`route_after_validation`** (P1.5): If `rule_violations` or `missing_info` are non-empty, routes to `negotiate` (terminates the turn). Otherwise proceeds to `orchestrate`.

2. **`route_after_review`** (P4.5): If violations exist and `review_retry_count == 0`, loops back to `itinerary_planning` for one re-plan attempt. Otherwise proceeds to `respond`.

## Key Design Patterns

### Factory Functions with Closures

All graph nodes use factory functions (e.g., `create_intent_node(llm)`) that inject dependencies via closures, keeping the node signatures compatible with LangGraph's `async def node(state) -> dict` contract.

### Parallel Execution

The orchestrate node groups skill agents by priority level. Agents at the same priority run concurrently via `asyncio.gather`. Between batches, results from completed agents (e.g., transport options) are injected into the context for downstream agents.

### Plugin Architecture

Skill agents are loaded dynamically from `.claude/skills/` via `LazyAgentRegistry`. Each skill directory contains a `SKILL.md` (metadata) and `script/agent.py` (implementation). Agents are only instantiated on first use.
