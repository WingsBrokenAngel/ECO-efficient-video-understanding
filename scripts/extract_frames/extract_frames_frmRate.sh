#!/bin/bash
#1:File address,2:frm_rate,3:save_path

EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` video frames/sec [size=256] save path"
  exit $E_BADARGS
fi


ffmpeg -i '$1' -vf scale=320:320 -qscale:v 2 -r $FRAMES  '$3_%4d.jpg'
