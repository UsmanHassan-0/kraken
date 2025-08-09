# Kraken UI + SDR Clean Repo

This is my **private, minimal version** of the KrakenSDR-DOA repo.  
It contains only the folders that I work on directly:

- `_sdr/`
- `_ui/`

---

## 🔥 Purpose

This repo is used to:

- Develop and test changes in `_sdr/` and `_ui/` independently
- Push code to GitHub
- Pull into a full KrakenSDR install (ADB or other systems)
- Replace only the updated code (not the full repo)

---

## 🛠️ How I Use This

1. Clone this repo
2. Work on `_sdr/` or `_ui/`
3. Push changes here
4. On the target system (with full KrakenSDR-DOA repo):
   - Copy updated `_sdr/` and `_ui/` folders into the full repo
   - Do **not** overwrite other system files

---

## 🧼 Cleaned via Git Filter-Repo

The following files/folders were removed from history using `git filter-repo`:

- `.github/`
- `.gitignore`
- `LICENSE`
- `README.md` (from original)
- `kill.sh`
- `gui_run.sh`
- `doc/`
- `util/`
- `_nodejs/`
- `.pre-commit-config.yaml`
- `pyproject.toml`
- `_ui/_web_interface/assets/`

### Clean Command:

```bash
git filter-repo --path ... --invert-paths --force
