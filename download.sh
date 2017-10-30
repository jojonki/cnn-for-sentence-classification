#!/bin/sh

DATA_DIR="./dataset/"
mkdir -p $DATA_DIR

download () {
  URL=$1
  FILE_NAME=$2

  if [ ! -f "$DATA_DIR$FILE_NAME" ]; then
    wget $URL$FILE_NAME -O $DATA_DIR/$FILE_NAME
  else
    echo "You've already downloaded $FILE_NAME dataset"
  fi
}


download "https://www.cs.cornell.edu/people/pabo/movie-review-data/" "rt-polaritydata.tar.gz"
download "http://nlp.stanford.edu/data/" "glove.6B.zip"
