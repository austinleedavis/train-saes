# %%
import os

import chess
import datasets
import plotly_express as px
import torch
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from src.model import SparseCrosscoder
from src.utils.chess_utils import uci_to_board, uci_to_pgn

PROBE_DEVICE = "cuda"
LLM_DEVICE = "cuda"
# %%

tok: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
    "austindavis/chessGPT2"
)
llm: GPT2LMHeadModel = (
    GPT2LMHeadModel.from_pretrained("austindavis/chessGPT2")
    .train(False)
    .requires_grad_(False)
    .to(LLM_DEVICE)
)
scc: SparseCrosscoder = (
    SparseCrosscoder.from_pretrained("models/last-v3-state_dict.pt")
    .train(False)
    .requires_grad_(False)
    .to(PROBE_DEVICE)
)
ds: datasets.Dataset = datasets.load_dataset(
    "austindavis/lichess-uci", "201301", split="train"
)


# %%
stoi = {k: v for v, k in enumerate(list("PNBRQKpnbrqk"), start=1)}
stoi[None] = 0
itos = {v: k for k, v in stoi.items()}


def board_to_piece_list(board: chess.Board):
    piece_map = board.piece_map()
    piece_list = [0] * 64
    for k, v in piece_map.items():
        piece_list[k] = stoi[v.symbol()]
    return torch.tensor(piece_list, dtype=torch.long)


def collate(record: dict, max_tokens: int = 1 + 3 * 40):
    # compute hidden states
    transcript = record["Transcript"]
    text_encoding = tok.encode_plus(
        transcript,
        return_tensors="pt",
        return_length=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        max_length=max_tokens,
        truncation=True,
    ).to(LLM_DEVICE)
    *_, hidden_states = llm.forward(**text_encoding, output_hidden_states=True).values()
    stacked_hidden_states = torch.stack(hidden_states[1:], dim=1).to(PROBE_DEVICE)

    # compute board states
    board_stack = uci_to_board(
        transcript, as_board_stack=True, map_function=board_to_piece_list
    )
    length = text_encoding.length.item()
    board_indices = [0] + [i // 3 for i in range(length - 1)]
    boards = [board_stack[i] for i in board_indices]
    boards = torch.stack(boards, dim=1).to(PROBE_DEVICE)
    return dict(
        model_activations_BLPD=stacked_hidden_states,
        transcript=transcript,
        boards=boards,
        text_encoding=text_encoding,
    )


# %%

############
# Setup Probe
############

# Use this line if training from scratch:
# probe = torch.nn.ModuleList(
# [torch.nn.Linear(scc.dict_size, len(stoi), bias=True) for _ in range(64)]
# ).to(PROBE_DEVICE)

# Use this block if loading from checkpoint:
CKPT_TO_LOAD = "models/probe-last-v3/probe-step-481995-loss-0.46329.pt"
ckpt = torch.load(CKPT_TO_LOAD, weights_only=False)
probe: torch.nn.ModuleList = ckpt["probe"]

print(probe)


# %%
def save_checkpoint(
    epoch, ds, step, losses, last_checkpoint_index, checkpoint_files, save_folder, probe
):
    global_step = len(ds) * epoch + step
    loss_val = "NA"
    if losses:
        loss_val = f"{losses[-1]['avg_loss']:.5f}"
    filename = os.path.join(save_folder, f"probe-step-{global_step}-loss-{loss_val}.pt")

    # cycle the checkpoint files so save only up to NUM_CHECKPOINTS
    this_checkpoint_index = (last_checkpoint_index + 1) % NUM_CHECKPOINTS
    if checkpoint_files[this_checkpoint_index] is not None:
        os.remove(checkpoint_files[this_checkpoint_index])
    checkpoint_files[this_checkpoint_index] = filename

    torch.save(dict(probe=probe, global_step=global_step), f=filename)
    return this_checkpoint_index


# %%
############
# Main Training Section
############

LOG_INTERVAL = 1000
CHECKPOINT_INTERVAL = 1000
GRADIENT_ACCUM_STEPS = 10
NUM_CHECKPOINTS = 3
POS_OFFSET = 0
POS_STEP_SIZE = 3
POS_COUNT = 24
NUM_EPOCHS = 1
MODEL_PATH = "models/probe-last-v3"
LOG_FILE = os.path.join(MODEL_PATH, "probe.log")

os.makedirs(MODEL_PATH, exist_ok=True)

class_counts = {"p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1, "none": 16}
class_weights = torch.tensor(
    [
        1.0 / class_counts[str(c[0]).lower()]
        for c in sorted(stoi.items(), key=lambda x: x[1])
    ]
)
class_weights = class_weights / class_weights.sum()
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(PROBE_DEVICE))

optimizer = torch.optim.AdamW(probe.parameters(), lr=0.00011371)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, NUM_EPOCHS * len(ds) // GRADIENT_ACCUM_STEPS
)

min_characters = (5 * POS_COUNT // 3) + 1

losses = []  # contains: (step, avg_loss)
checkpoint_files = [None] * NUM_CHECKPOINTS
last_checkpoint_index = -1
with open(LOG_FILE, "a") as log:
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epoch", leave=True, total=NUM_EPOCHS):
        ds = ds.shuffle()
        optimizer.zero_grad()
        running_loss = 0
        for step, record in enumerate(tqdm(ds, desc="Step", leave=True)):
            if len(record["Transcript"]) < min_characters:
                continue

            model_activations_BLPD, transcript, boards, text_encoding = collate(
                record
            ).values()
            features = scc.encode(
                model_activations_BLPD
            ).squeeze()  # eliminate batch dim
            del model_activations_BLPD  # free that memory fast

            predictions = torch.stack([p(features) for p in probe]).permute(0, 2, 1)
            selected_predictions = predictions[:, :, -POS_COUNT:][
                :, :, POS_OFFSET::POS_STEP_SIZE
            ]
            selected_boards = boards[:, -POS_COUNT:][:, POS_OFFSET::POS_STEP_SIZE]
            loss = loss_fn(selected_predictions, selected_boards)
            loss.backward()

            running_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if (step + 1) % LOG_INTERVAL == 0:
                avg_loss = running_loss / LOG_INTERVAL
                running_loss = 0
                global_step = len(ds) * epoch + step
                losses.append(
                    dict(
                        epoch=epoch,
                        step=step,
                        global_step=global_step,
                        avg_loss=avg_loss,
                    )
                )
                log.writelines(
                    [
                        f"|  Epoch: {1+epoch: 3}  |  Step: {1+step: 8,}  |  Average Loss: {avg_loss:.5f}  |  LR: {lr_scheduler.get_last_lr()[-1]:.8f}  |\n"
                    ]
                )
                log.flush()

            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                last_checkpoint_index = save_checkpoint(
                    epoch,
                    ds,
                    step,
                    losses,
                    last_checkpoint_index,
                    checkpoint_files,
                    MODEL_PATH,
                    probe,
                )

exit(1)
# %%
px.line(
    x=[d["global_step"] for d in losses],
    y=[d["avg_loss"] for d in losses],
    color=[d["epoch"] for d in losses],
    title="Loss Curve",
    labels=dict(x="Step", y="Loss"),
).show()
# %%

####################################
# EXPLORING PROBE BELOW
####################################


##################
# Setup probe exploration data
##################
record = ds[0]

model_activations_BLPD, transcript, boards, text_encoding = collate(record).values()
features = scc.encode(model_activations_BLPD).squeeze()  # eliminate batch dim
predictions = torch.stack([p(features) for p in probe]).permute(0, 2, 1)


# %%

##################
# Exploration results
##################

preds = predictions.detach().to("cpu").squeeze().softmax(1).view(8, 8, 13, -1)[:, :, 1:]
POS_OFFSET = 4
POS_STEP_SIZE = 3
fig = px.imshow(
    preds[:, :, :, POS_OFFSET::POS_STEP_SIZE].flip(0),
    facet_col=2,
    title=f"{POS_OFFSET=}",
    facet_col_wrap=6,
    animation_frame=3,
)


for i, annotation in enumerate(fig.layout.annotations):
    if "facet_col" in annotation["text"]:
        annotation["text"] = list(stoi.keys())[i]
fig.show()
# %%
px.imshow(
    predictions.detach()
    .to("cpu")
    .squeeze()
    .max(1, keepdim=True)
    .indices.view(8, 8, 1, -1)[:, :, :, POS_OFFSET::POS_STEP_SIZE]
    .flip(0),
    animation_frame=3,
    facet_col=2,
    zmin=0,
    zmax=12,
    color_continuous_scale="Rainbow",
)
# %%
print(uci_to_pgn(transcript))

# %%.


def label(val):
    return chess.UNICODE_PIECE_SYMBOLS.get(itos[int(val)], " ")


pred = predictions.detach().to("cpu").squeeze()
frame_tensor = (
    pred.max(1, keepdim=True)
    .indices.view(8, 8, 1, -1)[:, :, :, POS_OFFSET::POS_STEP_SIZE]
    .flip(0)
)

fig = px.imshow(
    frame_tensor,
    animation_frame=3,
    facet_col=2,
    zmin=0,
    zmax=30,
    color_continuous_scale="deep",
    height=800,
    x=list("abcdefgh"),
    y=list("87654321"),
)

for f in fig.frames:
    for i, z in enumerate(f.data):
        text = [[label(val) for val in row] for row in z["z"]]
        z["text"] = text
        z["texttemplate"] = "%{text}"
        z["textfont"] = {"color": "black", "size": 16}

for z in fig.data:
    text = [[label(val) for val in row] for row in z["z"]]
    z["text"] = text
    z["texttemplate"] = "%{text}"
    z["textfont"] = {"color": "black", "size": 16}


fig.show()

# %%
