clear

# Control
python src/app.py --east data/frozen_east_text_detection.pb --image images/example_01.jpg

# Start / End / Stock Loss
python src/app.py --east data/frozen_east_text_detection.pb --image images/go.png
python src/app.py --east data/frozen_east_text_detection.pb --image images/game.png
python src/app.py --east data/frozen_east_text_detection.pb --image images/game-cropped.png
python src/app.py --east data/frozen_east_text_detection.pb --image images/game-cropped-adjusted.png
python src/app.py --east data/frozen_east_text_detection.pb --image images/2-3.png
python src/app.py --east data/frozen_east_text_detection.pb --image images/player-names.png