r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0,
        lr_sched_factor=0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 256 
    hypers['seq_len'] = 60
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.0002
    hypers['lr_sched_factor'] = 0.002
    hypers['lr_sched_patience'] = 5
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq= "SCENE:\n IDC; Herzliya;\nACT I."  
    temperature = 0.46
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**<br>
  If we try to load the complete corpus into an array/tensor/matrix etc, we might run out of memory<br>
  due a enormous shape. furthermore, the most important feature of RNN is learning over time<br>
  hence we maintain and control the sequence size so that our network will maintain<br> 
  the connections between the beginning of the text and its end.<br>
  In addition, if inserting the complete corpus, we actually will force our<br>
  model to overfit due to learning through memorization.


"""

part1_q2 = r"""
**Your answer:**<br>
Multilayer gated recurrent unit model, has "memory".
therefore, in consists a memory gate which allows hidden state to either pass through time
and be remembered, or otherwise be forgotten and not to pass the hidden state.
this feature enables to model larger length of sequence.


"""

part1_q3 = r"""
**Your answer:**<br>
as opposed to previous assingments, shuffeling the order of words actually matter.<br>
their order gives the text its meaning thus shuffeling it will cause the model<br>
learn some arbitrary permutation of words thus produce some unmeaningful text.<br>
furthermore, shuffeling the batches meaning we detach the connections of the hidden statses
meaning we ruin the model's memory.
.
"""

part1_q4 = r"""
**Your answer:**<br>
1) The temperature addition meant to scale to logits in the softmax such that it will influence the output distribution.<br>
we tend to lower the temperature below 1 in order to change the output distribution into 'harder' distribution - decreasing the lower probabilities candidates and increasing the higher ones.
So, we keep decreasing it (to a point!) so that our model will likely to pick the best candidates, and not pick a random candidate with small but possible probability.<br>
By doing this, we assume that our model is confident on his predictions, which may cause the following in Q3.<br><br>

2)Higher temperatures will 'soften' the distribution by increasing the uniformness thus increasing randomness$\rightarrow$ less probable options might be chosen.<br>
if we keep increasing and increasing the temperature we will end up with completer random, unmeaningful and maybe gibrish text.<br><br>

3) We want to decrease our temperature, its true, but only to certain extent because when decreasing too much we cause a very repetitive and uninteresting text.
  we trying to achieve a production of diverse text, and by decreasing the Temperature too much, the probability "mass"<br>
  will be concentrated in a few high probable tokens.<br>
  this will cause the model selcet a number of candidates over and over again.
"""
# ==============

