sudo docker run --rm -it --gpus all \
    -v $(pwd)/:/app \
    pymarl \
    python3 src/main.py --config=transf_qmix --env-config=sc2
    # for spread, change with: python3 src/main.py --config=transf_qmix --env-config=mpe/spread
