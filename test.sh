#!/bin/bash

clear

LAYOUT="raw_line"
PADDING=0.2
CONFIDENCE=0.9
MIN_NMS=0.4

echo "Layout: $LAYOUT, Padding: $PADDING, Confidence: $CONFIDENCE, Minimum NMS: $MIN_NMS"

# Control
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -i images/example_01.jpg -t "OH. OK"

# Start / End / Stock Loss
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -i images/go.png -t "GO!" "Emberwyn" "CPU" "SONIC"
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -i images/game.png -t "GAME!" "Emberwyn" "SONIC"
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -i images/game-cropped-adjusted.png -t "GAME!"
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -i images/2-3.png -t "2 - 3" "Emberwyn" "SONIC" "CPU" "-1" "1"
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -i images/player-names.png -t "Zelda" "P1" "CPU" "Vs." "Emberwyn" "SONIC"