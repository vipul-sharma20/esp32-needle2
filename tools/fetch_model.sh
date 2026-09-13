#!/bin/sh
# Fetch the Needle-2 weights (13 MB) from Hugging Face into model/.
set -e
DEST="$(dirname "$0")/../model"
URL=https://huggingface.co/Cactus-Compute/needle-2/resolve/main/needle2.cact
mkdir -p "$DEST"
if [ -f "$DEST/needle2.cact" ]; then
    echo "$DEST/needle2.cact already present"
    exit 0
fi
echo "fetching needle2.cact ..."
curl -L --fail --progress-bar -o "$DEST/needle2.cact" "$URL"
echo "-> $DEST/needle2.cact  ($(du -h "$DEST/needle2.cact" | cut -f1))"
echo "next: python3 tools/pack_esp.py model/needle2.cact build/needle.nsp"
