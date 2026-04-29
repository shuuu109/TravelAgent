# State Reference

The global state is a `TypedDict` called `TravelGraphState` defined in `graph/state.py`. Every node reads from and writes to this shared state.

## State Layers

The state is organized into four conceptual layers:

### 1. Dialogue Layer

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `messages` | `list[BaseMessage]` | `add_messages` | Full conversation history. Messages are appended, never overwritten. |

### 2. Constraint Layer

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `hard_constraints` | `HardConstraints` | replace | Origin, destination, dates, pax |
| `soft_constraints` | `SoftConstraints` | replace | Hotel brands, airlines, seat, budget, other preferences |
| `rule_violations` | `List[RuleViolation]` | replace | Physical/temporal constraint violations |

### 3. Planning Layer

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `missing_info` | `List[str]` | replace | Hard constraint fields still needed from user |
| `current_plan` | `Dict[str, Any]` | replace | Generated itinerary structure |
| `transport_options` | `List[Dict]` | replace | Validated transport options (serialized `TravelOption`) |
| `travel_style` | `str` | replace | `"亲子"` / `"情侣"` / `"特种兵"` / `"普通"` |
| `travel_days` | `int` | replace | Computed from start/end dates |
| `poi_candidates` | `List[Dict]` | replace | Raw POI results from P2 |
| `daily_itinerary` | `List[Dict]` | replace | POIs grouped by day after clustering |
| `daily_routes` | `List[Dict]` | replace | TSP-optimized routes per day |
| `daily_restaurants` | `List[Dict]` | replace | Nearby restaurants per day |
| `rag_snippets` | `List[Dict]` | replace | Raw RAG retrieval documents |
| `rag_experience` | `ExperienceOutput` | replace | Structured tips and best_for from RAG |
| `rag_risks` | `RiskOutput` | replace | Structured risk warnings from RAG |
| `poi_descriptions` | `Dict[str, str]` | replace | POI name → 1-2 sentence description |
| `poi_search_hints` | `List[str]` | replace | LLM-generated POI search queries |
| `destination_best_season` | `str` | replace | Best travel season from city knowledge DB |
| `destination_transport_hubs` | `List[str]` | replace | Transport hubs from city knowledge DB |

### 4. Orchestration Layer

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `user_query` | `str` | replace | Raw user input text |
| `intent_data` | `Dict[str, Any]` | replace | Full intent node output |
| `intent_schedule` | `List[Dict]` | replace | Ordered list of agents to execute |
| `skill_results` | `List[Dict]` | `operator.add` | Results from skill agents (parallel-safe append) |
| `final_response` | `str` | replace | Final text response to user |
| `review_retry_count` | `int` | replace | Retry counter for P4.5 review loop (max 1) |

## Pydantic Data Models

### HardConstraints

```python
class HardConstraints(BaseModel):
    origin: Optional[str]       # Origin city
    destination: Optional[str]  # Destination city
    start_date: Optional[str]   # Departure date
    end_date: Optional[str]     # Return date
    pax: Optional[int] = 1      # Number of travelers
```

**`is_complete()`**: Returns `True` when origin, destination, and start_date are all set.

### SoftConstraints

```python
class SoftConstraints(BaseModel):
    hotel_brands: List[str]              # Preferred hotel brands
    airlines: List[str]                  # Preferred airlines
    seat_preference: Optional[str]       # Window, aisle, etc.
    budget_level: Optional[str]          # Economy, luxury, etc.
    other_preferences: Dict[str, Any]    # Extensible key-value store
```

### RuleViolation

```python
class RuleViolation(BaseModel):
    violation_type: str         # "distance_error", "time_conflict", etc.
    description: str            # Human-readable explanation
    suggestion: Optional[str]   # Recommended fix
```

### TravelOption

```python
class TravelOption(BaseModel):
    transport_type: str               # "高铁" | "飞机"
    transport_no: Optional[str]       # G1234 / CA1234
    departure_time: Optional[str]
    arrival_time: Optional[str]
    duration: Optional[str]
    departure_hub: Optional[str]      # Station or airport
    arrival_hub: Optional[str]        # Key field for accommodation
    price_range: Optional[str]
    is_recommended: bool = False
    data_source: str = "llm"          # "realtime" | "llm"
```

### RAG Output Models

```python
class ExperienceOutput(BaseModel):
    tips: List[str]       # Actionable travel tips with specific details
    best_for: List[str]   # Why this destination suits the travel style

class RiskOutput(BaseModel):
    risks: List[str]      # Each risk includes scenario + consequence + mitigation

class PoiTimeInfo(BaseModel):
    poi_name: str
    estimated_hours: float = 1.5
    best_period: str = "flexible"  # morning/afternoon/evening/flexible
```

## Reducer Semantics

- **`add_messages`**: LangGraph built-in — appends new messages to the list
- **`operator.add`**: Python list concatenation — parallel nodes can safely append to `skill_results`
- **replace** (default): Last writer wins — the newest value overwrites the old one
