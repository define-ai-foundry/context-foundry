# Introduction

> **Context Foundry is an open platform for constructing, maintaining, and communicating evidence-based Operational Context.**

## Key Takeaways

* Context Foundry complements existing interoperability standards and command-and-control ecosystems.
* It integrates observations from diverse sources into a continuously evolving **Operational Context**.
* It preserves evidence, provenance, and uncertainty to support trustworthy and explainable systems.
* It provides a foundation for state estimation, contextual reasoning, analytics, AI-assisted services, and multi-protocol publication.
* Its purpose is to support **shared situational understanding** across humans and machines.

---

## Why Context Foundry?

Modern operational ecosystems increasingly rely on interoperable information exchange. Standards such as SAPIENT, Cursor-on-Target, and other established interfaces enable systems to share observations, tracks, and reports across organisational and technological boundaries.

As interoperability continues to mature, the next opportunity lies beyond exchanging information.

How can observations from diverse sources be integrated into a coherent, trustworthy, and continuously evolving representation of the operational environment?

Context Foundry addresses this challenge by constructing and maintaining **Operational Context**—a shared, evidence-based representation of the operational environment that can be consumed by humans, command-and-control systems, analytics, and future reasoning services.

---

## From Observations to Understanding

Context Foundry is built around a simple idea:

> **Interoperability enables systems to exchange information. Operational Context enables them to develop shared situational understanding.**

Rather than processing observations as isolated messages, Context Foundry continuously refines an evidence-based representation of the operational environment while preserving provenance, uncertainty, and traceability.

This Operational Context provides a common foundation from which different consumers—human or machine—can derive understanding appropriate to their own mission and responsibilities.

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
        │
 ┌──────┼─────────┬──────────────┐
 ▼      ▼         ▼              ▼
Operators  C2 Systems  Analytics  Reasoning Services
```

---

## What Makes Context Foundry Different?

Context Foundry is not defined by a particular messaging standard, tracking algorithm, storage technology, or artificial intelligence framework.

Instead, it is built around a stable architectural abstraction: **Operational Context**.

This allows the platform to evolve alongside emerging technologies while remaining compatible with existing interoperability standards, operational workflows, and deployment environments.

The objective is not to replace existing command-and-control platforms or interoperability solutions, but to strengthen the quality, traceability, and consistency of the operational context shared between them.

---

## Core Concepts

The documentation introduces a small set of concepts that define the platform.

| Concept                 | Description                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Operational Context** | A continuously evolving, evidence-based representation of the operational environment.                           |
| **Observation**         | Information received from a sensor, system, or other trusted source.                                             |
| **Evidence**            | A coherent representation constructed from one or more related observations.           |
| **Reasoning Services**  | Components that derive additional insight from Operational Context using rules, analytics, AI, or other methods. |


> **These concepts remain stable even as implementations evolve.**

---

## Architecture at a Glance

Each architectural layer adds meaning while preserving the evidence beneath it.

```text
External Sources
        │
        ▼
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
        │
        ▼
Reasoning & Enrichment
        │
        ▼
 Publication
```

Operational Context is the central abstraction that connects every stage of the platform.

---

## Learn More

The documentation is organised into four complementary sections:

* **Concepts** introduce the fundamental ideas that define Context Foundry.
* **Architecture** explains how those concepts interact within the platform.
* **Developer Guide** describes how to extend and integrate Context Foundry.
* **Reference** provides configuration, APIs, schemas, and implementation details.

Readers interested in understanding the platform should begin with the Concepts section, starting with **Operational Context**.

---

## Guiding Principle

Every architectural decision in Context Foundry should contribute to one objective:

> **Construct and maintain trustworthy Operational Context that enables shared situational understanding while preserving the evidence that supports it.**
