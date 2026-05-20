"""Smoke test that instantiates representative blocks and runs a forward pass.

Run with ``python -m flax_blocks._smoke_test``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from . import (
    core_blocks as cb,
    attention_blocks as ab,
    transformer_blocks as tb,
    cnn_vision_blocks as vb,
    unet_diffusion_blocks as ub,
    gan_blocks as gb,
    vit_blocks as vit,
    sequence_blocks as sb,
    gnn_blocks as gnn,
    generative_blocks as gen,
    rl_blocks as rl,
    memory_retrieval_blocks as mem,
    embedding_blocks as emb,
    optimization_blocks as opt,
    multimodal_blocks as mm,
    efficient_blocks as eff,
    specialized_blocks as spc,
)


def _shape(x):
    if isinstance(x, jax.Array):
        return tuple(x.shape)
    return type(x).__name__


def main() -> None:
    rngs = nnx.Rngs(0)
    key = jax.random.key(42)

    print("== core ==")
    x = jnp.ones((2, 32, 32, 3))
    print(" ConvBlock        ", _shape(cb.ConvBlock(3, 16, rngs=rngs)(x)))
    print(" ResidualBlock    ",
          _shape(cb.ResidualBlock(3, 16, strides=2, rngs=rngs)(x)))
    print(" RMSNorm          ", _shape(nnx.RMSNorm(16, rngs=rngs)(jnp.ones((2, 5, 16)))))
    print(" InstanceNorm     ", _shape(cb.InstanceNorm(3, rngs=rngs)(x)))

    print("== attention ==")
    h = jnp.ones((2, 10, 64))
    print(" MultiHeadAttn    ", _shape(ab.MultiHeadAttention(64, 8, rngs=rngs)(h)))
    print(" CrossAttention   ",
          _shape(ab.CrossAttention(64, 32, 8, rngs=rngs)(h, jnp.ones((2, 12, 32)))))
    print(" LinearAttention  ", _shape(ab.LinearAttention(64, 8, rngs=rngs)(h)))
    print(" WindowAttention  ", _shape(ab.WindowAttention(64, 8, window=4, rngs=rngs)(h)))
    rope = ab.RotaryEmbedding(8)
    cos, sin = rope(10)
    print(" RoPE             ", cos.shape, sin.shape)
    print(" RelativePosBias  ",
          _shape(ab.RelativePositionBias(8, rngs=rngs)(10, 10)))
    print(" AttentionPool    ",
          _shape(ab.AttentionPooling(64, 4, rngs=rngs)(h)))

    print("== transformer ==")
    print(" Encoder block    ",
          _shape(tb.TransformerEncoderBlock(64, rngs=rngs)(h)))
    print(" Decoder block    ",
          _shape(tb.TransformerDecoderBlock(64, rngs=rngs)(h, h)))
    print(" SwiGLU           ", _shape(tb.SwiGLU(64, rngs=rngs)(h)))
    print(" GEGLU            ", _shape(tb.GEGLU(64, rngs=rngs)(h)))
    print(" MoE              ", _shape(tb.MixtureOfExperts(64, 4, 2, rngs=rngs)(h)))

    print("== cnn-vision ==")
    print(" Inception        ", _shape(vb.InceptionBlock(3, rngs=rngs)(x)))
    print(" SE               ", _shape(vb.SqueezeExcitation(3, rngs=rngs)(x)))
    print(" CBAM             ", _shape(vb.CBAM(3, rngs=rngs)(x)))
    print(" SPP              ", _shape(vb.SpatialPyramidPooling()(x)))
    feats = [jnp.ones((2, 32, 32, 16)), jnp.ones((2, 16, 16, 32)),
             jnp.ones((2, 8, 8, 64))]
    fpn_out = vb.FeaturePyramidNetwork([16, 32, 64], rngs=rngs)(feats)
    print(" FPN              ", [tuple(f.shape) for f in fpn_out])
    print(" ASPP             ", _shape(vb.ASPP(3, 16, rngs=rngs)(x)))
    print(" PixelShuffleUp   ", _shape(vb.PixelShuffleUpsample(3, rngs=rngs)(x)))
    print(" DeformConv2d     ", _shape(vb.DeformableConv2d(3, 16, rngs=rngs)(x)))
    tokens = jnp.ones((2, 64, 32))
    print(" DeformAttn       ",
          _shape(vb.DeformableAttention(32, 4, 4, rngs=rngs)(tokens, (8, 8))))

    print("== unet/diffusion ==")
    unet = ub.UNet(in_ch=4, out_ch=4, base=16, ch_mults=(1, 2), rngs=rngs)
    print(" UNet             ",
          _shape(unet(jnp.ones((1, 16, 16, 4)), jnp.array([5]))))
    print(" TimeEmbed        ",
          _shape(ub.TimestepMLP(64, rngs=rngs)(jnp.array([1, 2, 3]))))
    base_lin = nnx.Linear(8, 8, rngs=rngs)
    print(" LoRALinear       ",
          _shape(ub.LoRALinear(base_lin, rank=2, rngs=rngs)(jnp.ones((2, 8)))))
    print(" ZeroConv         ",
          _shape(ub.ZeroConv(3, 4, rngs=rngs)(x)))
    print(" IPAdapter        ",
          _shape(ub.IPAdapterCrossAttention(64, 32, 32, rngs=rngs)(
              h, jnp.ones((2, 5, 32)), jnp.ones((2, 5, 32)))))

    print("== gan ==")
    z = jnp.ones((2, 4, 4, 64))
    print(" Generator step   ", _shape(gb.GeneratorBlock(64, 32, rngs=rngs)(z)))
    print(" Discriminator    ", _shape(gb.DiscriminatorBlock(64, 32, rngs=rngs)(z)))
    print(" MinibatchStd     ", _shape(gb.MinibatchStdDev()(z)))
    w = jnp.ones((2, 256))
    print(" ModulatedConv2d  ",
          _shape(gb.ModulatedConv2d(64, 32, 3, 256, rngs=rngs)(z, w)))
    print(" StyleBlock       ",
          _shape(gb.StyleBlock(64, 32, 256, rngs=rngs)(z, w, key)))

    print("== vit ==")
    img = jnp.ones((2, 64, 64, 3))
    pe = vit.PatchEmbedding(3, 16, 96, rngs=rngs)
    seq = pe(img)
    print(" PatchEmbed       ", _shape(seq))
    print(" CLS              ", _shape(vit.CLSToken(96, rngs=rngs)(seq)))
    print(" Swin window-attn ",
          _shape(vit.SwinWindowAttention(96, 8, window=4, rngs=rngs)(
              jnp.ones((8, 16, 96)))))
    print(" MIM mask         ",
          _shape(vit.MaskedImageModeling(96, rngs=rngs).random_masking(seq, key)[0]))

    print("== sequence ==")
    seq_inp = jnp.ones((2, 6, 16))
    print(" RNNCell unroll   ",
          _shape(sb.run_recurrent(sb.RNNCell(16, 32, rngs=rngs), seq_inp)))
    print(" LSTMCell unroll  ",
          _shape(sb.run_recurrent(sb.LSTMCell(16, 32, rngs=rngs), seq_inp)))
    print(" GRUCell unroll   ",
          _shape(sb.run_recurrent(sb.GRUCell(16, 32, rngs=rngs), seq_inp)))
    print(" SSM              ",
          _shape(sb.StateSpaceModel(16, 8, rngs=rngs)(seq_inp)))
    print(" MambaBlock       ",
          _shape(sb.MambaBlock(16, 8, rngs=rngs)(seq_inp)))

    print("== gnn ==")
    feat = jnp.ones((6, 8))
    edges = jnp.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    print(" GraphConv        ", _shape(gnn.GraphConv(8, 16, rngs=rngs)(feat, edges)))
    print(" GAT              ",
          _shape(gnn.GraphAttention(8, 4, heads=2, rngs=rngs)(feat, edges)))

    print("== generative ==")
    vae = gen.VAE(in_ch=3, latent=16, channels=(16, 32), rngs=rngs)
    out, mu, lv = vae(jnp.ones((2, 16, 16, 3)), key)
    print(" VAE recon        ", _shape(out))
    print(" Coupling         ",
          _shape(gen.AffineCouplingLayer(8, rngs=rngs)(jnp.ones((4, 8)))[0]))
    sched = gen.DDPMScheduler(num_steps=10)
    xn, eps = sched.add_noise(jnp.ones((2, 16, 16, 3)), jnp.array([5, 5]), key)
    print(" DDPM noise       ", _shape(xn))

    print("== rl ==")
    pn = rl.PolicyNetwork(8, 3, discrete=True, rngs=rngs)
    print(" Policy logits    ",
          _shape(pn.logits_or_mean(jnp.ones((4, 8)))))
    rb = rl.ReplayBuffer(32)
    for _ in range(5):
        rb.push([0.0] * 4, 1, 1.0, [0.0] * 4, False)
    print(" Replay sample    ", [t.shape for t in rb.sample(2)])

    print("== memory ==")
    em = mem.ExternalMemory(num_slots=10, key_dim=8, value_dim=4, rngs=rngs)
    print(" ExternalMemory   ", _shape(em(jnp.ones((3, 8)))))
    cache = mem.KVCache(num_layers=1)
    cache.update(0, jnp.ones((1, 2, 4, 8)), jnp.ones((1, 2, 4, 8)))
    cache.update(0, jnp.ones((1, 1, 4, 8)), jnp.ones((1, 1, 4, 8)))
    print(" KVCache len      ", cache.length(0))

    print("== embedding ==")
    print(" SinusoidalPE     ",
          _shape(emb.SinusoidalPositionalEmbedding(20, 16)(jnp.ones((2, 12, 16)))))
    print(" InfoNCE          ",
          float(emb.info_nce(jnp.ones((8, 32)), jnp.ones((8, 32)) + 0.1)))

    print("== optimization ==")
    params = {"w": jnp.ones((4, 4)), "b": jnp.zeros((4,))}
    grads = {"w": jnp.ones((4, 4)) * 0.1, "b": jnp.ones((4,)) * 0.1}
    state = opt.lion_init(params)
    new_p, _ = opt.lion_update(grads, params, state, lr=1e-3)
    print(" Lion step        ", _shape(new_p["w"]))
    state2 = opt.adam_init(params)
    new_p, _ = opt.adam_update(grads, params, state2, lr=1e-3)
    print(" Adam step        ", _shape(new_p["w"]))
    sched = opt.cosine_warmup_schedule(2, 10, base_lr=1e-3)
    print(" Cosine lr@5      ", round(sched(5), 6))
    clipped, n = opt.clip_grad_norm(grads, 0.5)
    print(" GradNorm before  ", round(float(n), 4))

    print("== multimodal ==")
    pr = mm.PerceiverResampler(64, num_latents=8, depth=2, rngs=rngs)
    print(" PerceiverResamp  ", _shape(pr(jnp.ones((2, 50, 64)))))
    qf = mm.QFormer(dim=64, num_queries=8, num_heads=4, depth=2,
                    image_dim=128, llm_dim=256, rngs=rngs)
    print(" Q-Former         ", _shape(qf(jnp.ones((2, 50, 128)))))

    print("== efficient ==")
    lin = nnx.Linear(16, 32, rngs=rngs)
    q8 = eff.QuantizedLinearInt8.from_linear(lin, rngs=rngs)
    print(" Int8 Linear out  ", _shape(q8(jnp.ones((4, 16)))))
    q4 = eff.QuantizedLinear4bit.from_linear(lin, rngs=rngs)
    print(" 4-bit Linear out ", _shape(q4(jnp.ones((4, 16)))))
    lr = eff.LowRankLinear.from_linear(lin, rank=4, rngs=rngs)
    print(" LowRank Linear   ", _shape(lr(jnp.ones((4, 16)))))
    pruner = eff.MagnitudePruner(0.5)
    pruner.apply([lin])
    print(" Pruned sparsity  ",
          float(jnp.mean(lin.kernel.value == 0)))
    print(" TokenPruner      ",
          _shape(eff.TokenPruner(0.5)(jnp.ones((2, 8, 16)),
                                       jnp.arange(16, dtype=jnp.float32).reshape(2, 8))))

    print("== specialized ==")

    class Dyn(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            self.f = nnx.Linear(8, 8, rngs=rngs)
        def __call__(self, h: jax.Array, t: jax.Array) -> jax.Array:
            return jnp.tanh(self.f(h))

    print(" NeuralODE        ",
          _shape(spc.NeuralODE(Dyn(rngs=rngs), num_steps=4)(jnp.ones((2, 8)))))
    print(" FNOBlock         ",
          _shape(spc.FNOBlock(4, modes_h=4, modes_w=4, rngs=rngs)(
              jnp.ones((2, 16, 16, 4)))))
    print(" KANLayer         ",
          _shape(spc.KANLayer(8, 16, rngs=rngs)(jnp.ones((2, 8)))))
    caps = spc.CapsuleLayer(num_in=10, dim_in=8, num_out=5, dim_out=16, rngs=rngs)
    print(" CapsuleLayer     ", _shape(caps(jnp.ones((2, 10, 8)))))
    print(" SlotAttention    ",
          _shape(spc.SlotAttention(num_slots=4, dim=32, rngs=rngs)(
              jnp.ones((2, 50, 32)), key)))


if __name__ == "__main__":
    main()
