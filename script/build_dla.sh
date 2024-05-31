
echo "Build DLA Loadable loadable for int8"
mkdir -p model/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
${TRTEXEC} --onnx=model/lightNetV2-T4x-960x960-sparse-autolabel.onnx --useDLACore=0 --buildDLAStandalone --saveEngine=model/loadable/lightnet.int8.fp16chwin.fp16chwout.standalone.bin  --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear --int8 --fp16 --calib=model/lightNetV2-T4x-960x960-sparse-autolabel.EntropyV2-calibration.cache --precisionConstraints=obey --layerPrecisions="001_convolutional":fp16,"112_convolutional_lgx":fp16,"116_convolutional_lgx":fp16,"120_convolutional_lgx":fp16
${TRTEXEC} --onnx=model/lightNetV2-anonimization-sparse-320x320.onnx --useDLACore=1 --buildDLAStandalone --saveEngine=model/loadable/lightnet_320.int8.fp16chwin.fp16chwout.standalone.bin  --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear --int8 --fp16 --calib=model/lightNetV2-anonimization-sparse-320x320.EntropyV2-calibration.cache --precisionConstraints=obey --layerPrecisions="001_convolutional":fp16,"112_convolutional_lgx":fp16,"116_convolutional_lgx":fp16,"120_convolutional_lgx":fp16
