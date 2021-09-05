
for f in $(find $1 -name '*.tar' -type f)
do
  bf=$(basename $f)
  depth=$(echo $bf | sed -e s/[^0-9]//g)
  onnx="${f%%.*}.onnx"
  time python export_resnet_torchvision.py --in_path $f --out_path $onnx --depth $depth
done
