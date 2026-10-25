#!/usr/bin/env bash
# Build the on-device transcriber CLI with the Xcode toolchain (never bare `swift build`).
# Output: labeling/transcriber/.build/transcribe
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_dir="$here/.build"
mkdir -p "$out_dir"

# Embedded Info.plist with NSSpeechRecognitionUsageDescription. Note: when run from a
# terminal, TCC attributes Speech Recognition to the parent (responsible) process, so
# this alone does not make the SFSpeechRecognizer fallback promptable; the binary
# refuses that path unless already authorized. The default SpeechTranscriber engine
# needs no grant.
cat > "$out_dir/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key>
  <string>blue.catbird.intel.labeling.transcribe</string>
  <key>CFBundleName</key>
  <string>transcribe</string>
  <key>NSSpeechRecognitionUsageDescription</key>
  <string>Transcribes labeling voice notes entirely on this Mac.</string>
</dict>
</plist>
PLIST

xcrun --sdk macosx swiftc \
  -O \
  -parse-as-library \
  -swift-version 5 \
  -target arm64-apple-macosx26.0 \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "$out_dir/Info.plist" \
  "$here/transcribe.swift" \
  -o "$out_dir/transcribe"

# Ad-hoc sign so TCC can key any Speech Recognition grant to a stable identity.
codesign --force --sign - --identifier blue.catbird.intel.labeling.transcribe "$out_dir/transcribe"

echo "$out_dir/transcribe"
