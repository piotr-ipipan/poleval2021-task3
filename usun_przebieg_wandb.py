#########
#USUWANIE PRZEBIEGOW Z WANDB

import wandb
api = wandb.Api()
run = api.run("piotrb/sh-v5/31k9x5v0")
run.delete()