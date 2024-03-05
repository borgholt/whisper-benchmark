To run the test:
```
docker build . -t whisper-benchmark:latest
docker run -it --gpus device=0 whisper-benchmark
```
