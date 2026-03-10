#!/bin/bash
set -e

BINARY_NAME="thor-201909061227-Linux64"
ZIP_URL="http://s3-us-west-2.amazonaws.com/ai2-thor/builds/${BINARY_NAME}.zip"
INSTALL_DIR="${HOME}/.ai2thor/releases/${BINARY_NAME}"
ZIP_PATH="/tmp/${BINARY_NAME}.zip"

if [ -f "${INSTALL_DIR}/${BINARY_NAME}" ]; then
    echo "AI2-THOR binary already exists at ${INSTALL_DIR}/${BINARY_NAME}. Skipping download."
    exit 0
fi

echo "Downloading AI2-THOR binary from ${ZIP_URL} ..."
wget "${ZIP_URL}" -O "${ZIP_PATH}"

echo "Extracting to ${INSTALL_DIR} ..."
mkdir -p "${INSTALL_DIR}"
python -c "
import zipfile
with zipfile.ZipFile('${ZIP_PATH}') as z:
    z.extractall('${INSTALL_DIR}/')
"
chmod +x "${INSTALL_DIR}/${BINARY_NAME}"

rm -f "${ZIP_PATH}"
echo "Done. Binary installed at ${INSTALL_DIR}/${BINARY_NAME}"
