BASEDIR="$1"

time python resizeIms.py \
  --input_dir "$BASEDIR/full/val/" \
  --output_dir "$BASEDIR/161-jpeg-75/val/" \
  --quality 75 --ext jpg

time python resizeIms.py \
  --input_dir "$BASEDIR/full/val/" \
  --output_dir "$BASEDIR/161-jpeg-95/val/" \
  --quality 95 --ext jpg

time python resizeIms.py \
  --input_dir "$BASEDIR/full/val/" \
  --output_dir "$BASEDIR/161-png/val/" \
  --ext png
