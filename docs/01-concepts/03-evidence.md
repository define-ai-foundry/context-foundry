# Evidence

> **Evidence is a coherent representation constructed from one or more related observations. It provides the factual foundation from which Operational Context is built.**

## Key Takeaways

* Evidence is constructed by combining related observations.
* Multiple observations may contribute to a single piece of Evidence.
* Evidence preserves links to its contributing observations.
* Evidence represents what is currently known about an entity or event.
* Operational Context is built from Evidence rather than directly from observations.

---

## Why Evidence?

Individual observations rarely provide a complete picture of the operational environment.

Different sensors observe different aspects of the same entity or event.

A radar may determine position and velocity.

An electro-optical sensor may provide classification.

A passive RF sensor may detect emissions.

Each observation contributes useful information, but none provides a complete understanding on its own.

Evidence combines these related observations into a coherent representation while preserving their individual contributions.

---

## Example

An airborne object is observed by multiple sensors.

```text
Observation A
Radar
• Position
• Velocity

Observation B
EO Sensor
• Fixed-wing aircraft
• Confidence: Moderate

Observation C
Passive RF
• UHF emissions detected
```

These observations are associated as describing the same entity.

Together they form a single piece of Evidence.

```text
Evidence

Entity: E-104

Supported by:
✓ Radar observation
✓ EO observation
✓ RF observation

Current estimate:
• Position
• Velocity
• Classification
• Emissions
```

The original observations remain available.

Evidence establishes the relationship between them and provides a coherent representation that can be used to construct Operational Context.

---

## Constructing Evidence

Evidence is produced by associating related observations.

The techniques used to perform this association may vary depending on the operational domain and implementation.

Examples include:

* multi-sensor tracking,
* data association,
* correlation,
* event fusion, or
* other domain-specific fusion techniques.

Context Foundry does not prescribe how Evidence is constructed.

It only defines the role that Evidence plays within the architecture.

---

## Evidence Is Not...

| Evidence is not...  | Because...                                                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| A raw observation   | It combines information from one or more related observations.                                                              |
| A message           | The same Evidence may be constructed from observations received through many different protocols.                           |
| Operational Context | Evidence describes entities or events. Operational Context establishes their broader operational meaning and relationships. |

---

## Relationship to Operational Context

Evidence provides the building blocks from which Operational Context is constructed.

```text
Observations
       │
       ▼
Association & Fusion
       │
       ▼
Evidence
       │
       ▼
Operational Context
```

Operational Context enriches Evidence by placing it within a wider operational picture.

This may include relationships between entities, temporal evolution, mission relevance, reasoning, and information published to downstream consumers.

---

## Summary

Evidence is a coherent representation constructed from one or more related observations.

It preserves the connection to its contributing observations while providing a stable, implementation-independent foundation for Operational Context.

By separating observations from Evidence, Context Foundry allows different fusion techniques and association algorithms to evolve independently without changing the architecture or the meaning of Operational Context.
