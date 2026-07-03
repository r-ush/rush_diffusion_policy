#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

XACRO_FILE="$DIR/hand.urdf.xacro"
URDF_FILE="$DIR/hand.urdf"

if [ ! -f "$XACRO_FILE" ]; then
  echo "❌ Xacro file not found: $XACRO_FILE"
  exit 1
fi

echo "🔄 Converting $XACRO_FILE → $URDF_FILE"
ros2 run xacro xacro "$XACRO_FILE" -o "$URDF_FILE"

if [ $? -eq 0 ]; then
  echo "✅ Successfully generated: $URDF_FILE"
else
  echo "❌ Failed to generate URDF"
fi

