#!/usr/bin/env bash
set -euo pipefail

# Installer for the renamed Kraken repo (gui/ + daq/ layout).
# Run this script from within the repo root.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="kraken"
ARCH="$(uname -m)"

echo "[1/8] Install system dependencies"
sudo apt update
sudo apt -y install build-essential git cmake libusb-1.0-0-dev lsof libzmq3-dev clang php-cli nodejs gpsd libfftw3-bin libfftw3-dev wget

echo "[2/8] Build librtlsdr"
cd "$HOME"
if [[ ! -d librtlsdr ]]; then
  git clone https://github.com/krakenrf/librtlsdr
fi
cd librtlsdr
sudo cp rtl-sdr.rules /etc/udev/rules.d/rtl-sdr.rules
mkdir -p build && cd build
cmake ../ -DINSTALL_UDEV_RULES=ON
make
sudo ln -sf "$PWD/src/rtl_test" /usr/local/bin/kraken_test
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-dvb_usb_rtl28xxu.conf

echo "[3/8] Build DSP dependency (kfr on x86_64, Ne10 on aarch64)"
cd "$HOME"
if [[ "$ARCH" == "x86_64" ]]; then
  if [[ ! -d kfr ]]; then git clone https://github.com/krakenrf/kfr; fi
  cd kfr && mkdir -p build && cd build
  cmake -DENABLE_CAPI_BUILD=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release ..
  make
  sudo cp "$PWD"/lib/* /usr/local/lib
  sudo mkdir -p /usr/include/kfr
  sudo cp "$PWD"/../include/kfr/capi.h /usr/include/kfr
elif [[ "$ARCH" == "aarch64" ]]; then
  if [[ ! -d Ne10 ]]; then git clone https://github.com/krakenrf/Ne10; fi
  cd Ne10 && mkdir -p build && cd build
  cmake -DNE10_LINUX_TARGET_ARCH=aarch64 -DGNULINUX_PLATFORM=ON -DCMAKE_C_FLAGS="-mcpu=native -Ofast -funsafe-math-optimizations" ..
  make
  sudo cp "$PWD"/modules/libNE10.a /usr/local/lib
fi
sudo ldconfig

echo "[4/8] Install Miniforge and create conda env"
cd "$HOME"
if [[ "$ARCH" == "x86_64" ]]; then
  MINIFORGE=Miniforge3-Linux-x86_64.sh
else
  MINIFORGE=Miniforge3-Linux-aarch64.sh
fi
if [[ ! -d "$HOME/miniforge3" ]]; then
  if [[ ! -x "$HOME/$MINIFORGE" ]]; then
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE"
    chmod +x "$MINIFORGE"
  fi
  ./"$MINIFORGE" -b
fi
export PATH="$HOME/miniforge3/bin:$PATH"
eval "$(conda shell.bash hook)"
# Normalize condarc to avoid duplicate keys
printf "auto_activate: false\n" > "$HOME/.condarc"
conda config --set auto_activate false
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.9.7
fi
conda activate "$ENV_NAME"

echo "[5/8] Python dependencies"
# Base numeric stack (safe to re-run)
conda install -y scipy==1.9.3 numba==0.56.4 configparser pyzmq pandas orjson matplotlib requests scikit-image scikit-rf gitpython "blas=*=mkl" -c conda-forge

# Ensure pip deps are present (idempotent)
declare -a PIP_PKGS=(
  "dash==1.20.0"
  "dash-bootstrap-components==0.13.1"
  "dash_devices==0.1.3"
  "quart==0.17.0"
  "quart_compress==0.2.1"
  "werkzeug==2.0.2"
  "plotly==5.23.0"
  "pyargus"
  "gpsd-py3"
)

missing=()
for pkg in "${PIP_PKGS[@]}"; do
  if ! python - <<PY >/dev/null 2>&1
import pkg_resources, sys
pkg_resources.require("${pkg}")
PY
  then
    missing+=("${pkg}")
  fi
done

if (( ${#missing[@]} )); then
  pip install "${missing[@]}"
fi

echo "[6/8] Ensure DAQ build paths exist"
mkdir -p "$REPO_ROOT/daq/Firmware/data_control" "$REPO_ROOT/daq/Firmware/logs"

echo "[7/8] Build DAQ core"
cd "$REPO_ROOT/daq/Firmware/daq_core"
cp "$HOME/librtlsdr/build/src/librtlsdr.a" .
cp "$HOME/librtlsdr/include/rtl-sdr.h" .
cp "$HOME/librtlsdr/include/rtl-sdr_export.h" .
make

echo "[8/8] Done"
echo "Repo root: $REPO_ROOT"
echo "Conda env: $ENV_NAME"
echo "Start the system with: cd $REPO_ROOT && ./kraken_start.sh"
