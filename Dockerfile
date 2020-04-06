FROM python:buster

WORKDIR /usr/src/app
ENV TZ=America/Los_Angeles
# ENV ZSH_CUSTOM=/root/.oh-my-zsh/custom

# Python Dependencies
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install nms opencv-contrib-python imutils pytesseract pillow

# Develop Tools
RUN pip install coloredlogs
RUN apt-get update && apt-get install -y zsh vim
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN git clone https://github.com/reobin/typewritten.git /root/.oh-my-zsh/custom/themes/typewritten
RUN ln -s "/root/.oh-my-zsh/custom/themes/typewritten/typewritten.zsh-theme" "/root/.oh-my-zsh/custom/themes/typewritten.zsh-theme"
ADD .devcontainer/.zshrc /root

# Tesseract
RUN echo "deb https://notesalexp.org/tesseract-ocr/buster/ buster main" >> /etc/apt/sources.list
RUN wget -O - https://notesalexp.org/debian/alexp_key.asc | apt-key add -
RUN apt-get update && apt-get install -y apt-transport-https tesseract-ocr

# VOLUME [ "/usr/src/data" ]

CMD [ "/bin/zsh" ]