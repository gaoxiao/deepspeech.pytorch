docker run --runtime=nvidia \
-it \
-e NVIDIA_VISIBLE_DEVICES=2 \
-v ${PWD}:/code \
--rm \
--shm-size=2g \
--name DS \
deepspeech2.docker \
/bin/bash