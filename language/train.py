# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
# %%
model = Transformer.from_config(
    n_layer=1,
    d_model=1024,
    mlp="blp",
    d_hidden=1024*3,
    normalization=None,
    n_head=8,
    noise=0.33,
)

model.summary()
# %%
model.fit(log=True, epochs=5, wd=1, batch_size=128)
# %%
model.generate("Once upon a time, ", max_length=100)
# %%
model.push_to_hub(f"TinyStories-1-1024-inl")
# %%

