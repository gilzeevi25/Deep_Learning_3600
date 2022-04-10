r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=100,
        seq_len=100,
        h_dim=256,
        n_layers=3,
        dropout=0.05,
        learn_rate=0.003,
        lr_sched_factor=0.9,
        lr_sched_patience=4,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======

    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
"""

part1_q2 = r"""
**Your answer:**
"""

part1_q3 = r"""
**Your answer:**
"""

part1_q4 = r"""
**Your answer:**
"""
# ==============

