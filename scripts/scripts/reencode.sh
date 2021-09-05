reencode_dir() {
  echo "$1"
  for f in /lfs/1/ddkang/blazeit/data/svideo/$1/$2/*.mp4 ;
  do
    bn=$(basename -- $f)
    ffmpeg -i $f \
      -vf scale=854:480 \
      -c:v libx264 -preset slow -crf 20 -c:a copy \
      /lfs/1/ddkang/vision-inf/data/noscope/480p/$1/$bn
  done
}

reencode_dir "jackson-town-square" "2017-12-17"
reencode_dir "amsterdam" "2017-04-12"
reencode_dir "archie-day" "2018-04-11"
reencode_dir "taipei-hires" "2017-04-13"
reencode_dir "venice-rialto" "2018-01-20"
reencode_dir "venice-grand-canal" "2018-01-20"
