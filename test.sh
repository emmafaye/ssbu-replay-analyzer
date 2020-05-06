#!/bin/bash

clear

# for i in {0..35}
# do
    MAX_ANGLE=10
    MIN_NMS=0.1
    CONFIDENCE=0.9995
    PADDING=0.07
    LAYOUT=9
    # MIN_NMS=`echo 0.01 \* $i | bc`
    # PADDING=`echo 0.01 \* $i | bc`

    echo "Layout: $LAYOUT, Padding: $PADDING, Confidence: $CONFIDENCE, Minimum NMS: $MIN_NMS, Max Angle: $MAX_ANGLE"

    echo "--------------------------------------------------------------"

    # Reset Logging & Matches
    truncate -s 0 results.log
    rm assets/matches/*.*

    # Control
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/example_01.jpg -t "OH" "OK"

    # Start / End / Stock Loss
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/go.png -t "GO!" "Emberwyn" "CPU" "SONIC"
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/game.png -t "GAME!" "Emberwyn" "SONIC"
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/game-cropped-adjusted.png -t "GAME!"
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/2-3.png -t "Emberwyn" "SONIC" "CPU"
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT assets/images/player-names.png -t "Zelda" "P1" "CPU" "Vs." "Emberwyn" "SONIC"

    echo "--------------------------------------------------------------"

    python src/score.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT results.log | tee -a scores.log
# done