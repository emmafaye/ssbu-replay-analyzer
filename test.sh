#!/bin/bash

clear

MAX_ANGLE=10
MIN_NMS=0.1
CONFIDENCE=0.9995
PADDING=0.07
LAYOUT=11

while getopts "a:n:c:p:l:o" opt; do
    case "$opt" in
        a ) MAX_ANGLE="$OPTARG" ;;
        n ) MIN_NMS="$OPTARG" ;;
        c ) CONFIDENCE="$OPTARG" ;;
        p ) PADDING="$OPTARG" ;;
        l ) LAYOUT="$OPTARG" ;;
        o ) OUTPUT="$OPTARG" ;;
    esac
done

execute_test() {
    echo "Layout: $LAYOUT, Padding: $PADDING, Confidence: $CONFIDENCE, Minimum NMS: $MIN_NMS, Max Angle: $MAX_ANGLE"

    echo "——————————————————————————————————————————————————————————————————————————————————————————————"

    # Reset Logging, Matches & Output
    truncate -s 0 results.log
    rm -f assets/matches/*.*
    rm -f assets/output/*.*

    # Control
    # unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -o assets/images/example_01.jpg -t "OH" "OK"

    # Start / End / Stock Loss
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -o assets/images/go.png -t "GO!" "Emberwyn" "CPU" "SONIC"
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -o assets/images/game.png -t "GAME!" "Emberwyn" "SONIC"
    # unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -o assets/images/game-cropped-adjusted.png -t "GAME!"
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -o assets/images/2-3.png -t "Emberwyn" "SONIC" "CPU"
    unbuffer python src/ocr.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT -o assets/images/player-names.png -t "Zelda" "P1" "CPU" "Vs." "Emberwyn" "SONIC"

    echo "——————————————————————————————————————————————————————————————————————————————————————————————"

    python src/score.py -a $MAX_ANGLE -n $MIN_NMS -c $CONFIDENCE -p $PADDING -l $LAYOUT results.log | tee -a scores.log
}

# Runs multiple tests of layouts 0 through 13
if [ $LAYOUT == -1 ]; then
    for i in {0..13}; do
        LAYOUT=$i
        execute_test
    done
# Runs multiple tests of Minimum NMS 0.00 through 0.35
elif [ $MIN_NMS == -1 ]; then
    for i in {0..35}; do
        MIN_NMS=`echo 0.01 \* $i | bc`
        execute_test
    done
# Runs multiple tests of Minimum NMS 0.00 through 0.20
elif [ $PADDING == -1 ]; then
    for i in {0..20}; do
        PADDING=`echo 0.01 \* $i | bc`
        execute_test
    done
else
    execute_test
fi