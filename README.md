# train-saes

# Environment Variables
```sh
WANDB_API_KEY=... # alternatively, log in to wandb within the container.
NTFY_TOPIC=<your_topic_here> # the topic to which you will publish/subscribe notifications
```

  - `WANDB_API_KEY`: If WandB logger is used, this is the key or login via the WandB CLI
  - `NTFY_TOPIC`: The topic to which you will publish/subscribe notifications
  - `DATA_ROOT`: Root directory for local copy of hidden state vector dataset