clear

# Control
python src/analyze.py --east data/frozen_east_text_detection.pb --image images/example_01.jpg

# Start / End / Stock Loss
python src/analyze.py --east data/frozen_east_text_detection.pb --image images/go.png  --padding 1
python src/analyze.py --east data/frozen_east_text_detection.pb --image images/game.png --padding 1
python src/analyze.py --east data/frozen_east_text_detection.pb --image images/game-cropped.png --padding 1
python src/analyze.py --east data/frozen_east_text_detection.pb --image images/game-cropped-adjusted.png --padding 1
python src/analyze.py --east data/frozen_east_text_detection.pb --image images/2-3.png --padding 1