#!/bin/bash

clear

LAYOUT="raw_line"
PADDING=0.1
CONFIDENCE=0.9
MIN_NMS=0.4

echo "Layout: $LAYOUT, Padding: $PADDING, Confidence: $CONFIDENCE, Minimum NMS: $MIN_NMS"

echo "--------------------------------------------------------------"

# Reset Logging & Matches
truncate -s 0 results.log
rm assets/matches/*.*

# Control
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/example_01.jpg -t "OH" "OK" >> results.log

# Start / End / Stock Loss
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/go.png -t "GO!" "Emberwyn" "CPU" "SONIC" >> results.log
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/game.png -t "GAME!" "Emberwyn" "SONIC" >> results.log
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/game-cropped-adjusted.png -t "GAME!" >> results.log
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/2-3.png -t "2 - 3" "Emberwyn" "SONIC" "CPU" "-1" "1" >> results.log
python src/ocr.py -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/player-names.png -t "Zelda" "P1" "CPU" "Vs." "Emberwyn" "SONIC" >> results.log

echo "--------------------------------------------------------------"

python src/score.py results.log