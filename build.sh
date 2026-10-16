#!/bin/zsh
set -e
cd "$(dirname "$0")"
APP=build/monti-fondo.app
rm -rf build && mkdir -p $APP/Contents/MacOS
swiftc -O -parse-as-library App.swift -o $APP/Contents/MacOS/monti-fondo
cp Info.plist $APP/Contents/
mkdir -p $APP/Contents/Resources && cp AppIcon.icns $APP/Contents/Resources/
codesign --force --sign - $APP
echo "OK: $APP"
