# Graph Nodes

All nodes follow the **factory function** pattern: `create_*_node(llm, ...)` returns an `async def node(state: TravelGraphState) -> dict` closure. Dependencies (LLM, registries, memory managers) are injected via the closure.

## Node Pipeline

| Priority | Node | Module | Description |
|----------|------|--------|-------------|
| P1 | `intent` | `intent_node.py` | Intent recognition and entity extraction |
| P1.5 | `validate_constraints` | `node.py` | Amap MCP-powered spatiotemporal feasibility check |
| P1.5b | `negotiate` | `node.py` | Terminates the turn with violation feedback |
| P2 | `orchestrate` | `orchestrate_node.py` | Parallel skill dispatch by priority |
| P3 | `itinerary_planning` | `itinerary_planning_node.py` | POI clustering, TSP routing, restaurant search |
| P3.5 | `poi_enrich` | `poi_enrich_node.py` | RAG-based POI experience descriptions |
| P4 | `accommodation` | `accommodation_node.py` | Hotel recommendations via RollingGo MCP |
| P4.5 | `itinerary_review` | `itinerary_review_node.py` | Self-check with optional retry loop |
| P5 | `respond` | `respond_node.py` | Final structured response generation |

---

## P1: Intent Node (`intent_node.py`)

**Factory**: `create_intent_node(llm)`

Converts user natural language input into a structured intent schedule. Uses the LLM with a detailed system prompt to:

1. Classify intent into one of 6 categories: `itinerary_planning`, `information_query`, `preference`, `memory_query`, `rag_knowledge`, `event_collection`
2. Extract key entities: origin, destination, dates, pax, travel style
3. Generate an `agent_schedule` — a priority-ordered list of skill agents to execute

**Writes to state**: `intent_data`, `intent_schedule`, `hard_constraints`, `soft_constraints`, `travel_style`, `travel_days`, `poi_search_hints`, `destination_best_season`, `destination_transport_hubs`

The node also loads skill descriptions via `SkillLoader` and looks up city knowledge from `CityKnowledgeDB` for season and transport hub data.

---

## P1.5: Validate Constraints (`node.py::validate_rule_constraints`)

An **agentic validation** node that spawns a ReAct sub-agent equipped with Amap MCP tools. The sub-agent:

1. Queries real driving/transit distance and time between origin and destination
2. Checks if the user's date range can accommodate the round-trip travel time
3. Returns a JSON verdict with `is_valid`, `violation_type`, `description`, and `suggestion`

**Writes to state**: `rule_violations`

**Routing**: If violations or missing info exist → `negotiate` (turn ends). Otherwise → `orchestrate`.

---

## P1.5b: Negotiate Constraints (`node.py::negotiate_constraints`)

Formats `rule_violations` and `missing_info` into a human-readable response and writes it to `final_response`. The turn ends here — the user must address the violations or provide missing information.

---

## P2: Orchestrate Node (`orchestrate_node.py`)

**Factory**: `create_orchestrate_node(registry, memory_manager)`

Executes the `intent_schedule` from P1. Groups agents by priority level and runs same-priority agents in parallel via `asyncio.gather`. Between batches, enriches the shared context with transport and accommodation results for downstream agents.

**Key behavior**:
- Looks up agents from `LazyAgentRegistry` by name
- Calls each agent's `run(context)` method
- Collects results into `skill_results` (uses `operator.add` reducer for parallel-safe appending)

**Writes to state**: `skill_results`, `poi_candidates`, `rag_snippets`, `rag_experience`, `rag_risks`, `transport_options`

---

## P3: Itinerary Planning Node (`itinerary_planning_node.py`)

**Factory**: `create_itinerary_planning_node(llm)`

The core planning node that transforms raw POI candidates into a day-by-day itinerary:

1. **POI Selection**: Scores and selects top POIs based on Amap rating, RAG affinity, and travel style
2. **LLM Time Estimation**: Batch-queries the LLM for estimated visit duration and best time-of-day per POI
3. **Daily Clustering**: Assigns POIs to days using geographic proximity (K-Means style)
4. **TSP Route Optimization**: Orders POIs within each day to minimize travel time using the Amap distance matrix
5. **Restaurant Search**: Finds nearby restaurants around each day's geographic centroid via Amap

**Writes to state**: `daily_itinerary`, `daily_routes`, `daily_restaurants`

---

## P3.5: POI Enrich Node (`poi_enrich_node.py`)

**Factory**: `create_poi_enrich_node(llm)`

Post-retrieval augmentation step. For each POI in the itinerary, queries the RAG knowledge base to retrieve relevant travel experiences, then uses the LLM to distill a 1-2 sentence description.

**Writes to state**: `poi_descriptions` (dict mapping POI name → description)

---

## P4: Accommodation Node (`accommodation_node.py`)

**Factory**: `create_accommodation_node(llm, memory_manager)`

Queries the RollingGo MCP server for hotel recommendations based on:
- Arrival hub (from transport options or city knowledge fallback)
- User preferences (budget level, hotel brands)
- Travel dates

**Writes to state**: `current_plan` (accommodation section)

---

## P4.5: Itinerary Review Node (`itinerary_review_node.py`)

**Factory**: `create_itinerary_review_node()`

Self-check node that validates the generated itinerary against physical constraints (e.g., daily travel time budgets, opening hours). If violations are found and `review_retry_count == 0`, routes back to P3 for one re-planning attempt.

**Writes to state**: `rule_violations`, `review_retry_count`

---

## P5: Respond Node (`respond_node.py`)

**Factory**: `create_respond_node(llm)`

Generates the final user-facing response by assembling:

1. **Daily route blocks**: For each day, lists POIs with descriptions, transit info, and time slots
2. **Restaurant recommendations**: Appended to each day's section
3. **Travel tips**: Filtered from `rag_experience.tips` (excludes generic POI recommendations)
4. **Risk warnings**: From `rag_risks.risks`
5. **Accommodation summary**: From `current_plan`
6. **LLM fallback**: If no structured data is available, uses the LLM to generate a freeform response

**Writes to state**: `final_response`

---

## Utility Nodes (in `node.py`)

### `extract_hard_constraints`

Extracts origin, destination, dates, and pax from user messages using structured output (`llm.with_structured_output(HardConstraints)`). Merges new values with existing constraints using a "new value wins, else keep old" strategy.

### `enrich_soft_constraints`

Loads user preferences from long-term memory and maps them to `SoftConstraints` fields (hotel brands, airlines, seat preference, budget level). Unknown preference types go into `other_preferences`.

### `plan_itinerary`

Legacy planning node (superseded by `itinerary_planning_node` in the current pipeline).
