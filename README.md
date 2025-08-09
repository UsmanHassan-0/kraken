# KrakenSDR DoA — Private Minimal Code Drop
[![Status](https://img.shields.io/badge/status-active-brightgreen)](#)
[![Scope](https://img.shields.io/badge/scope-_sdr%20%2B%20_ui_-blue)](#)
[![Style](https://img.shields.io/badge/commits-elite%20%2F%20surgical-black)](#)
[![History](https://img.shields.io/badge/history-filtered-important)](#)

> **Purpose:** This is my **private, minimal** KrakenSDR-DoA repo containing only the parts I actively maintain:
>
> - `_sdr/` (receiver + signal processing)
> - `_ui/` (Dash web UI)
>
> I develop here, push to my private remote, then **copy these two folders** into a machine that has the **official KrakenSDR-DoA** repo. Nothing else in the upstream project is modified by this repo.

---

## Table of Contents
- [Why this repo exists](#why-this-repo-exists)
- [Repo layout](#repo-layout)
- [Workflow (dev → sync → verify)](#workflow-dev--sync--verify)
- [Quick start](#quick-start)
- [Sync script](#sync-script)
- [Standards](#standards)
- [History hygiene (filter-repo)](#history-hygiene-filter-repo)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)

---

## Why this repo exists
The full Kraken project includes CI, scripts, docs, assets, etc. I only edit signal processing and the UI. This repo:
- keeps my focus **laser-sharp** on `_sdr` and `_ui`;
- avoids noise/merge churn with upstream;
- makes it trivial to **drop-in replace** just the code I changed.

> 🔎 **Note on history:** This repo uses `git filter-repo` to remove everything except `_sdr/` and `_ui/`. See [History hygiene](#history-hygiene-filter-repo) and `.git-cleanup-log.md`.

---

## Repo layout
.
├── _sdr/
│ ├── _receiver/
│ └── _signal_processing/
└── _ui/
└── _web_interface/
├── app.py
├── callbacks/
├── views/
└── (optional) assets/

yaml
Copy

---

## Workflow (dev → sync → verify)

```mermaid
flowchart LR
  A[Edit in private repo (_sdr/_ui)] --> B[Commit & push to private remote]
  B --> C[On target box with official Kraken repo]
  C --> D[Copy _sdr & _ui over official repo]
  D --> E[Run app & verify UI/SDR behavior]
