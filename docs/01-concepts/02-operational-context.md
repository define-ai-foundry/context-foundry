# Operational Context

> **Operational Context is the continuously evolving, evidence-based representation of an operational environment. It provides a shared foundation for humans and machines to develop situational understanding.**

## Key Takeaways

* Operational Context is the central abstraction of Context Foundry.
* It is constructed from evidence, not individual messages.
* It preserves provenance, uncertainty, and relationships.
* It is independent of communication protocols and implementation technologies.
* Every major component of Context Foundry contributes to constructing, maintaining, enriching, or communicating Operational Context.

---

## Why Operational Context?

Modern operational systems can exchange information more effectively than ever before through established interoperability standards and interfaces.

Exchanging information, however, is only one step towards shared understanding.

Observations must be correlated, validated, related to one another, and interpreted within their operational environment before they become truly useful for decision support.

Operational Context provides a common, evidence-based representation that brings these elements together.

Rather than asking *"What message was received?"*, Operational Context asks *"What is our current understanding of the operational environment, and what evidence supports that understanding?"*

---

## From Observations to Operational Context

Operational Context is built through progressive refinement.

```text
Physical Reality
        │
        ▼
 Observations
        │
        ▼
   Evidence
        │
        ▼
Operational Context
        │
        ▼
Shared Situational Understanding
```

Each stage adds meaning while preserving the information contributed by earlier stages.

The objective is not to replace observations, but to organise them into a coherent representation that supports both human and machine reasoning.

---

## Example

Consider an unidentified airborne object.

* A radar reports its position and velocity.
* A passive RF sensor detects radio emissions.
* An electro-optical sensor classifies it as a fixed-wing UAV.
* An operator reports unusual flight behaviour.

Each observation contributes to evidence of physical reality.

Operational Context relates these observations to the same entity while preserving:

* their origin,
* timestamps,
* confidence,
* uncertainty, and
* relationships.

Different consumers may then use the same Operational Context in different ways.

* A command-and-control system may receive a Cursor-on-Target event.
* Another system may consume a SAPIENT publication.
* A reasoning service may analyse behavioural patterns.
* A human operator may view the complete operational picture.

Although the representations differ, they all originate from the same Operational Context.

---

## Characteristics

Operational Context has several defining characteristics.

### Continuously Evolving

Operational Context is continuously refined as new observations become available.

### Evidence-based

Every assessment should be supported by evidence with known provenance.

### Explainable

Consumers should be able to determine not only what the platform believes, but also why it reached that conclusion.

### Shared

Operational Context provides a common representation that supports multiple consumers without requiring each to maintain its own interpretation of the operational environment.

### Technology Independent

Operational Context is an architectural concept rather than a software implementation.

It remains valid regardless of messaging standards, storage technologies, tracking algorithms, reasoning frameworks, or programming languages.

---

## Operational Context Is Not...

Operational Context is sometimes confused with related concepts.

It is important to distinguish them.

| It is not...            | Because...                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| A message               | Multiple messages may contribute to the same Operational Context.                                                  |
| A track                 | Tracks represent entities; Operational Context also captures evidence, relationships, provenance, and uncertainty. |
| A database              | Databases store information. Operational Context represents meaning.                                               |
| Artificial Intelligence | AI may enrich or interpret Operational Context, but it does not define it.                                         |

---

## Relationship to Other Concepts

Operational Context connects the core concepts used throughout Context Foundry.

```text
Observations
      │
      ▼
Evidence
      │
      ▼
Operational Context
      │
      ├────────► Publication
      ├────────► Reasoning Services
      ├────────► Analytics
      └────────► Human Operators
```

The following concept documents describe these relationships in more detail:

* **Observation**
* **Evidence**
* **Reasoning Services**
* **Publication**

---

## Summary

Operational Context is the central architectural abstraction of Context Foundry.

It represents the platform's continuously evolving understanding of the operational environment, built from evidence and designed to support shared situational understanding across humans, command-and-control systems, analytics, and future reasoning services.

Everything in Context Foundry exists to build, maintain, enrich, or communicate Operational Context.
