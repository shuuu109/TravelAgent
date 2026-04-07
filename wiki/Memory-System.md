# Memory System

Aligo implements a **two-layer memory** architecture that separates session-scoped dialogue context from persistent user data.

## Overview

```
┌─────────────────────────────────────────────┐
│              MemoryManager                  │
│  (context/memory_manager.py)                │
│                                             │
│  ┌───────────────────┐ ┌─────────────────┐  │
│  │  ShortTermMemory  │ │ LongTermMemory  │  │
│  │  (in-memory list) │ │ (JSON file)     │  │
│  │                   │ │                 │  │
│  │  • Last 10 turns  │ │ • Preferences   │  │
│  │  • Auto-eviction  │ │ • Chat history  │  │
│  │  • Session-scoped │ │ • Trip records  │  │
│  │                   │ │ • Statistics    │  │
│  └───────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────┘
```

## Short-Term Memory

**Class**: `ShortTermMemory` (`context/short_term_memory.py`)

An in-memory sliding window of the current session's dialogue.

| Property | Value |
|----------|-------|
| Storage | Python list (in-process) |
| Capacity | 10 turns (20 messages) |
| Eviction | FIFO — oldest messages drop when capacity is exceeded |
| Lifetime | Current session only |

### API

```python
memory.short_term.add_message(role, content, metadata=None)
memory.short_term.get_recent_context(n_turns=5)  # Returns list of message dicts
memory.short_term.get_context_string(n_turns=5)   # Returns formatted string
memory.short_term.clear()
memory.short_term.get_statistics()
```

### Message Format

```python
{
    "role": "user",           # or "assistant"
    "content": "...",
    "timestamp": "2025-01-01T12:00:00",
    "metadata": {}
}
```

## Long-Term Memory

**Class**: `LongTermMemory` (`context/long_term_memory.py`)

Persistent user data stored as a JSON file at `data/memory/{user_id}.json`.

| Property | Value |
|----------|-------|
| Storage | JSON file on disk |
| Lifetime | Permanent (across sessions) |
| Data types | Preferences, chat history, trip history, statistics |

### Storage Structure

```json
{
  "user_id": "default_user",
  "created_at": "2025-01-01T00:00:00",
  "updated_at": "2025-01-01T12:00:00",
  "preferences": [
    {"type": "home_location", "value": "Shanghai"},
    {"type": "budget_level", "value": "economy"},
    {"type": "hotel_brands", "value": ["Hilton", "Marriott"]}
  ],
  "chat_history": [
    {"role": "user", "content": "...", "timestamp": "...", "session_id": "abc123"}
  ],
  "trip_history": [
    {"trip_id": "trip_1", "origin": "Shanghai", "destination": "Beijing", ...}
  ],
  "statistics": {
    "total_trips": 5,
    "total_messages": 120,
    "frequent_destinations": {"Beijing": 3, "Hangzhou": 2}
  }
}
```

### API

```python
# Preferences
memory.long_term.save_preference(pref_type, value)
memory.long_term.get_preference()               # Returns dict of all preferences
memory.long_term.get_preference("budget_level")  # Returns specific preference
memory.long_term.add_hotel_brand("Hilton")       # Append to list preference
memory.long_term.add_airline("Air China")

# Chat history
memory.long_term.add_chat_message(role, content, session_id)
memory.long_term.get_chat_history(limit=10, session_id=None)

# Trip history
memory.long_term.save_trip_history(trip_info)
memory.long_term.get_trip_history(limit=10)
memory.long_term.get_frequent_destinations(top_n=5)

# Maintenance
memory.long_term.clear_history()  # Keeps preferences
memory.long_term.delete_all()     # Deletes the JSON file
```

## MemoryManager

**Class**: `MemoryManager` (`context/memory_manager.py`)

Unified facade that coordinates both layers.

### Key Methods

| Method | Description |
|--------|-------------|
| `add_message(role, content)` | Writes to both short-term and long-term storage |
| `get_full_context()` | Returns combined dict of both layers |
| `get_context_for_agent(long_term_summary)` | Formatted string for agent prompts |
| `get_long_term_summary_async(max_messages)` | LLM-powered summarization of past sessions |
| `end_session()` | Clears short-term memory |

### LLM Summarization

When an LLM model is available, `get_long_term_summary_async()` builds a prompt from chat history and trip records (excluding the current session), then asks the LLM to summarize in under 200 characters. This summary is injected into the intent node's system message as `【历史会话总结】`.

## Data Migration

`LongTermMemory._migrate_data()` handles backward compatibility:
- Converts old dict-format preferences to list format
- Fixes nested preference bugs from earlier versions
- Ensures all required fields exist
