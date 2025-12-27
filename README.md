# Autonomous Vision Agent

An autonomous real-time vision system that adapts its perception over time using memory, forgetting, selective attention, and self-calibrating confidence.

---

## Overview

This project explores how a vision system can move beyond static object detection and behave as an **adaptive perceptual agent**.

Instead of relying on fixed thresholds or manual control, the system continuously regulates how it perceives the world by learning from experience.

The agent:
- remembers objects it has seen before
- forgets objects that are no longer relevant
- suppresses attention on familiar, stable objects
- adapts its confidence dynamically based on novelty
- preserves learned experience across runs

---

## Key Features

- Real-time object detection using YOLOv8  
- Memory-based object identity (known vs unknown)  
- Hybrid memory decay based on time and stability  
- Selective attention suppression for familiar objects  
- Adaptive confidence threshold based on scene novelty  
- Long-term memory persistence across sessions  
- Live webcam operation with behavioral overlays  

---

## System Architecture

The system operates as a closed perception loop:

```
SEE → REMEMBER → FORGET → DECIDE → ADAPT → CONTINUE
```

---

## Project Structure

```
app/
  detector.py        # Object detection (YOLOv8)
cognition/
  embedding.py       # Visual representation
  similarity.py      # Identity matching
  memory.py          # Memory, stability, decay
  persistence.py     # Long-term memory persistence
webcam.py            # Autonomous agent loop
requirements.txt
README.md
```

---

## How the System Works

1. Objects are detected from a live webcam feed.
2. Visual embeddings are extracted for each detection.
3. Each object is matched against stored memory.
4. Objects are classified as known or unknown.
5. Memory entries decay over time using hybrid rules.
6. Familiar and stable objects receive reduced attention.
7. The system adapts its confidence threshold automatically.
8. Learned memory is saved and restored across executions.

All behavior emerges from internal system state rather than user control.

---

## Running the Project

```bash
python webcam.py
```

Press `q` to exit.

---

## Design Philosophy

This project intentionally avoids:
- fixed perception parameters
- manual runtime tuning
- frontend or user interface controls

The webcam window and overlays serve as an **instrument panel** for observing autonomous behavior, not a traditional UI.

---

## Notes

- Learned memory is stored locally and excluded from version control.
- Each execution contributes to the system’s long-term experience.
- The repository represents a completed system.
- Future extensions should be implemented in separate projects.

---

## Status

**Project complete.**

This repository is intentionally finalized and maintained as a reference implementation of an autonomous perceptual agent.
