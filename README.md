# Kraken SDR / Heimdall DAQ Workspace

This is my own version of kraken software with amendments of the original software

## Quick install

1) Clone and enter the repo (example in home dir):
   ```bash
   cd ~
   git clone https://github.com/UsmanHassan-0/kraken
   cd kraken
   ```
2) Run the installer from a shell with no conda env active:
   ```bash
   ./install_kraken.sh
   ```
   It will install Miniforge, create the `kraken` env, build SDR deps, and install Python packages.
3) Start the stack from the repo root:
   ```bash
   ./kraken_start.sh
   ```
   (Stops with `./kraken_stop.sh`.)
