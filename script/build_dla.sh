
echo "Build DLA Loadable loadable for int8"
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
${TRTEXEC} --onnx=model/lightNetV2-T4x-960x960-sparse-autolabel.onnx --useDLACore=0 --buildDLAStandalone --saveEngine=model/loadable/lightnet.int8.int8chwin.fp16chwout.standalone.bin  --inputIOFormats=int8:dla_linear --outputIOFormats=fp16:dla_linear --int8 --fp16 --calib=model/lightNetV2-T4x-960x960-sparse-autolabel.EntropyV2-calibration.cache --precisionConstraints=obey --layerPrecisions="112_convolutional_lgx":fp16,"116_convolutional_lgx":fp16,"120_convolutional_lgx":fp16
