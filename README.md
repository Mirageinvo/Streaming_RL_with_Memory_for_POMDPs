### Streaming RL with Memory for POMDPs

##### Here are command lines to run train/test for different architectures:
```
pip3 install -r requirements.txt
```

Stream-x:
```
python3 stream_ac_minigrid_no_memory.py
```

Stream-x + LSTM:
```
python3 stream_ac_minigrid_lstm.py
```

Batch PPO:
```
python3 batch_ppo_no_memory.py
```

Batch PPO + LSTM:
```
python3 batch_ppo_lstm_memory.py
```