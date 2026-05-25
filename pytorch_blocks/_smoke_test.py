"""Smoke-test that instantiates representative blocks and runs a forward pass.

Run with ``python -m pytorch_blocks._smoke_test``. It is *not* a unit-test
suite - just enough to catch obvious shape/import bugs. Each section prints
its output shape so you can visually verify dimensions.
"""

from __future__ import annotations

import torch

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
    return tuple(x.shape) if torch.is_tensor(x) else type(x).__name__


def main() -> None:
    torch.manual_seed(0)

    print("== core ==")
    x = torch.randn(2, 3, 32, 32)
    print(" ConvBlock        ", _shape(cb.ConvBlock(3, 16)(x)))
    print(" ResidualBlock    ", _shape(cb.ResidualBlock(3, 16, stride=2)(x)))
    print(" RMSNorm          ", _shape(cb.RMSNorm(16)(torch.randn(2, 5, 16))))

    print("== attention ==")
    h = torch.randn(2, 10, 64)
    print(" MultiHeadAttn    ", _shape(ab.MultiHeadAttention(64, 8)(h)))
    print(" CrossAttention   ", _shape(ab.CrossAttention(64, 32, 8)(h, torch.randn(2, 12, 32))))
    print(" LinearAttention  ", _shape(ab.LinearAttention(64, 8)(h)))
    print(" WindowAttention  ", _shape(ab.WindowAttention(64, 8, window=4)(h)))
    rope = ab.RotaryEmbedding(8)
    cos, sin = rope(10, h.device)
    print(" RoPE             ", cos.shape, sin.shape)

    print("== transformer ==")
    print(" Encoder block    ", _shape(tb.TransformerEncoderBlock(64)(h)))
    print(" Decoder block    ", _shape(tb.TransformerDecoderBlock(64)(h, h)))
    print(" SwiGLU           ", _shape(tb.SwiGLU(64)(h)))
    print(" MoE              ", _shape(tb.MixtureOfExperts(64, 4, 2)(h)))

    print("== cnn-vision ==")
    print(" Inception        ", _shape(vb.InceptionBlock(3)(x)))
    print(" SE               ", _shape(vb.SqueezeExcitation(3)(x)))
    print(" CBAM             ", _shape(vb.CBAM(3)(x)))
    print(" SPP              ", _shape(vb.SpatialPyramidPooling()(x)))
    feats = [torch.randn(2, 16, 32, 32), torch.randn(2, 32, 16, 16),
             torch.randn(2, 64, 8, 8)]
    print(" FPN out shapes   ", [tuple(f.shape) for f in vb.FeaturePyramidNetwork([16, 32, 64])(feats)])
    print(" ASPP             ", _shape(vb.ASPP(3, 16)(x)))
    print(" PixelShuffleUp   ", _shape(vb.PixelShuffleUpsample(3)(x)))
    print(" DeformConv2d     ", _shape(vb.DeformableConv2d(3, 16)(x)))
    tokens = torch.randn(2, 64, 32)
    print(" DeformAttn       ", _shape(vb.DeformableAttention(32, 4, 4)(tokens, (8, 8))))

    print("== unet/diffusion ==")
    unet = ub.UNet(in_ch=4, out_ch=4, base=16, ch_mults=(1, 2))
    print(" UNet             ", _shape(unet(torch.randn(1, 4, 16, 16),
                                             torch.tensor([5]))))
    print(" TimeEmbed        ", _shape(ub.TimestepMLP(64)(torch.tensor([1, 2, 3]))))
    base = torch.nn.Linear(8, 8)
    print(" LoRALinear       ", _shape(ub.LoRALinear(base, rank=2)(torch.randn(2, 8))))
    print(" ZeroConv         ", _shape(ub.ZeroConv2d(3, 4)(x)))
    print(" IPAdapter        ", _shape(ub.IPAdapterCrossAttention(64, 32, 32)(h,
        torch.randn(2, 5, 32), torch.randn(2, 5, 32))))

    print("== gan ==")
    z = torch.randn(2, 64, 4, 4)
    print(" Generator step   ", _shape(gb.GeneratorBlock(64, 32)(z)))
    print(" Discriminator    ", _shape(gb.DiscriminatorBlock(64, 32)(z)))
    print(" MinibatchStd     ", _shape(gb.MinibatchStdDev()(z)))
    w = torch.randn(2, 256)
    print(" ModulatedConv2d  ", _shape(gb.ModulatedConv2d(64, 32, 3, 256)(z, w)))
    print(" StyleBlock       ", _shape(gb.StyleBlock(64, 32, 256)(z, w)))
    print(" EqualLinear      ", _shape(gb.EqualLinear(64, 128)(torch.randn(2, 64))))
    print(" EqualConv2d      ", _shape(gb.EqualConv2d(64, 32, 3)(z)))

    print("== vit ==")
    img = torch.randn(2, 3, 64, 64)
    pe = vit.PatchEmbedding(3, 16, 96)
    seq = pe(img)
    print(" PatchEmbed       ", _shape(seq))
    print(" CLS              ", _shape(vit.CLSToken(96)(seq)))
    print(" Swin window-attn ", _shape(vit.SwinWindowAttention(96, 8, window=4)(
        torch.randn(8, 16, 96))))
    print(" MIM              ", _shape(vit.MaskedImageModeling(96).random_masking(seq)[0]))

    print("== sequence ==")
    seq_inp = torch.randn(2, 6, 16)
    print(" RNNCell unroll   ", _shape(sb.run_recurrent(sb.RNNCell(16, 32), seq_inp)))
    print(" LSTMCell unroll  ", _shape(sb.run_recurrent(sb.LSTMCell(16, 32), seq_inp)))
    print(" GRUCell unroll   ", _shape(sb.run_recurrent(sb.GRUCell(16, 32), seq_inp)))
    print(" SSM              ", _shape(sb.StateSpaceModel(16, 8)(seq_inp)))
    print(" MambaBlock       ", _shape(sb.MambaBlock(16, 8)(seq_inp)))

    print("== gnn ==")
    n = 6
    feat = torch.randn(n, 8)
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    print(" GraphConv        ", _shape(gnn.GraphConv(8, 16)(feat, edges)))
    print(" GAT              ", _shape(gnn.GraphAttention(8, 4, heads=2)(feat, edges)))

    print("== generative ==")
    vae = gen.VAE(in_ch=3, latent=16, channels=(16, 32))
    out, mu, lv = vae(torch.randn(2, 3, 16, 16))
    print(" VAE recon        ", _shape(out))
    print(" Coupling         ", _shape(gen.AffineCouplingLayer(8)(torch.randn(4, 8))[0]))
    sched = gen.DDPMScheduler(num_steps=10)
    xn, eps = sched.add_noise(torch.randn(2, 3, 16, 16), torch.tensor([5, 5]))
    print(" DDPM noise       ", _shape(xn))

    print("== rl ==")
    pn = rl.PolicyNetwork(8, 3, discrete=True)
    print(" Policy logits    ", _shape(pn(torch.randn(4, 8)).logits))
    rb = rl.ReplayBuffer(32)
    for _ in range(5):
        rb.push([0.0] * 4, 1, 1.0, [0.0] * 4, False)
    print(" Replay sample    ", [t.shape for t in rb.sample(2)])

    print("== memory ==")
    em = mem.ExternalMemory(num_slots=10, key_dim=8, value_dim=4)
    print(" ExternalMemory   ", _shape(em(torch.randn(3, 8))))
    cache = mem.KVCache(num_layers=1)
    cache.update(0, torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8))
    cache.update(0, torch.randn(1, 4, 1, 8), torch.randn(1, 4, 1, 8))
    print(" KVCache len      ", cache.length(0))

    print("== embedding ==")
    print(" Sinusoidal PE    ", _shape(emb.SinusoidalPositionalEmbedding(20, 16)(
        torch.randn(2, 12, 16))))
    print(" InfoNCE          ", emb.info_nce(torch.randn(8, 32), torch.randn(8, 32)).item())

    print("== optimization ==")
    m = torch.nn.Linear(8, 8)
    Lion = opt.Lion(m.parameters(), lr=1e-3)
    out = m(torch.randn(2, 8)).sum()
    out.backward()
    Lion.step()
    print(" Lion step ok")
    sch = opt.cosine_warmup_scheduler(Lion, warmup_steps=2, total_steps=10)
    sch.step(); sch.step(); sch.step()
    print(" Cosine warmup lr ", Lion.param_groups[0]["lr"])

    print("== multimodal ==")
    pr = mm.PerceiverResampler(64, num_latents=8, depth=2)
    print(" PerceiverResamp  ", _shape(pr(torch.randn(2, 50, 64))))
    qf = mm.QFormer(dim=64, num_queries=8, num_heads=4, depth=2,
                    image_dim=128, llm_dim=256)
    print(" Q-Former         ", _shape(qf(torch.randn(2, 50, 128))))

    print("== efficient ==")
    lin = torch.nn.Linear(16, 32)
    q8 = eff.QuantizedLinearInt8.from_linear(lin)
    print(" Int8 Linear out  ", _shape(q8(torch.randn(4, 16))))
    q4 = eff.QuantizedLinear4bit.from_linear(lin)
    print(" 4-bit Linear out ", _shape(q4(torch.randn(4, 16))))
    lr = eff.LowRankLinear.from_linear(lin, rank=4)
    print(" LowRank Linear   ", _shape(lr(torch.randn(4, 16))))
    pruner = eff.MagnitudePruner(0.5); pruner.apply([lin])
    print(" Pruned sparsity  ", (lin.weight == 0).float().mean().item())
    print(" TokenPruner      ", _shape(eff.TokenPruner(0.5)(
        torch.randn(2, 8, 16), torch.randn(2, 8))))

    print("== specialized ==")
    dyn = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh())
    class Dyn(torch.nn.Module):
        def __init__(self): super().__init__(); self.f = torch.nn.Linear(8, 8)
        def forward(self, h, t): return torch.tanh(self.f(h))
    print(" NeuralODE        ", _shape(spc.NeuralODE(Dyn(), num_steps=4)(torch.randn(2, 8))))
    print(" FNOBlock         ", _shape(spc.FNOBlock(4, modes_h=4, modes_w=4)(
        torch.randn(2, 4, 16, 16))))
    print(" KANLayer         ", _shape(spc.KANLayer(8, 16)(torch.randn(2, 8))))
    caps = spc.CapsuleLayer(num_in=10, dim_in=8, num_out=5, dim_out=16)
    print(" CapsuleLayer     ", _shape(caps(torch.randn(2, 10, 8))))
    print(" SlotAttention    ", _shape(spc.SlotAttention(num_slots=4, dim=32)(
        torch.randn(2, 50, 32))))


if __name__ == "__main__":
    main()
