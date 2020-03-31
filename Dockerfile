FROM python:buster

WORKDIR /usr/src/app

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install imutils
RUN pip install numpy
RUN pip install pytesseract
RUN pip install argparse
RUN pip install opencv-python

# VOLUME [ "/usr/src/data" ]

# COPY . .

CMD [ "/bin/bash" ]
#ENTRYPOINT [ "/bin/bash" ]