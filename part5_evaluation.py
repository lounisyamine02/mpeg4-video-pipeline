

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import zlib
import pickle

from part1_preprocessing import preprocess
from part2_iframe import decode_iframe,encode_all_iframe, decode_all_iframe, get_quant_matrix
from part3_pframe import encode_video, decode_pframe, MACRO_SIZE
from part4_entropy import entropy_encode, compute_original_size


#  5a -- Metriques

def compute_metrics(frames_bgr, encoded_video, bin_path="output/video_compressed.bin"):
    original_size = compute_original_size(frames_bgr)
    bin_size      = os.path.getsize(bin_path) if os.path.exists(bin_path) else 1
    ratio         = original_size / bin_size
    i_count       = sum(1 for e in encoded_video if e['type'] == 'I')
    p_count       = sum(1 for e in encoded_video if e['type'] == 'P')

    print(f"\n-- Metriques Partie 5 --")
    print(f"  Taille originale  : {original_size:,} octets")
    print(f"  Taille .bin       : {bin_size:,} octets")
    print(f"  Ratio compression : {ratio:.2f}x")
    print(f"  I-frames          : {i_count} | P-frames : {p_count}")

    return {
        'original_size': original_size,
        'bin_size':      bin_size,
        'ratio':         ratio,
        'i_count':       i_count,
        'p_count':       p_count,
    }




def compare_ratio_quant(preprocessed_frames, quant_factors=None):
   
    if quant_factors is None:
        quant_factors = [1, 5, 10, 20, 30, 50]

    ratios    = []
    n_frames  = min(5, len(preprocessed_frames))   
    orig_size = sum(
        preprocessed_frames[i]['shape'][0] * preprocessed_frames[i]['shape'][1] * 3
        for i in range(n_frames)
    )

    print("\n  Calcul ratio et Fq")
    for fq in quant_factors:
        encoded_list = [
            encode_all_iframe(preprocessed_frames[i], quant_factor=fq)
            for i in range(n_frames)
        ]
        raw_bytes        = pickle.dumps(encoded_list)
        compressed_bytes = zlib.compress(raw_bytes, level=6)
        ratio            = orig_size / len(compressed_bytes)
        ratios.append(ratio)
        print(f"    Fq={fq:2d} -> ratio={ratio:.2f}x")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(quant_factors, ratios, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(quant_factors, ratios, alpha=0.15, color='blue')
    ax.set_xlabel("Facteur de Quantification (Fq)", fontsize=12)
    ax.set_ylabel("Ratio de Compression", fontsize=12)
    ax.set_title("Ratio de Compression vs Facteur de Quantification", fontsize=13)
    ax.grid(True, alpha=0.3)
    for fq, r in zip(quant_factors, ratios):
        ax.annotate(f"{r:.1f}x", (fq, r), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("output/part5_ratio_vs_quant.png", dpi=120)
    plt.show()
    print("  Figure sauvegardee : output/part5_ratio_vs_quant.png")



def compare_ratio_gop(preprocessed_frames, gop_sizes=None, quant_factor=10):
  
    if gop_sizes is None:
        gop_sizes = [1, 2, 3, 5, 10]

    n_frames  = min(10, len(preprocessed_frames))
    ratios    = []
    i_counts  = []
    p_counts  = []
    orig_size = sum(
        preprocessed_frames[i]['shape'][0] * preprocessed_frames[i]['shape'][1] * 3
        for i in range(n_frames)
    )

    print("\n  Calcul ratio et GOP size")
    for gop in gop_sizes:
        enc_video, _ = encode_video(
            preprocessed_frames[:n_frames],
            quant_factor=quant_factor,
            gop_size=gop
        )
        raw_bytes        = pickle.dumps(enc_video)
        compressed_bytes = zlib.compress(raw_bytes, level=6)
        ratio   = orig_size / len(compressed_bytes)
        i_count = sum(1 for e in enc_video if e['type'] == 'I')
        p_count = sum(1 for e in enc_video if e['type'] == 'P')
        ratios.append(ratio)
        i_counts.append(i_count)
        p_counts.append(p_count)
        print(f"    GOP={gop:2d} -> ratio={ratio:.2f}x  (I={i_count}, P={p_count})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(gop_sizes, ratios, 'ro-', linewidth=2, markersize=8)
    ax1.fill_between(gop_sizes, ratios, alpha=0.15, color='red')
    ax1.set_xlabel("Taille du GOP", fontsize=12)
    ax1.set_ylabel("Ratio de Compression", fontsize=12)
    ax1.set_title("Ratio de Compression vs Taille GOP", fontsize=13)
    ax1.grid(True, alpha=0.3)
    for g, r in zip(gop_sizes, ratios):
        ax1.annotate(f"{r:.1f}x", (g, r), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    x     = np.arange(len(gop_sizes))
    width = 0.35
    ax2.bar(x - width/2, i_counts, width, label='I-frames', color='steelblue')
    ax2.bar(x + width/2, p_counts, width, label='P-frames', color='coral')
    ax2.set_xlabel("Taille du GOP", fontsize=12)
    ax2.set_ylabel("Nombre de frames", fontsize=12)
    ax2.set_title("I-frames vs P-frames par GOP", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(g) for g in gop_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("output/part5_ratio_vs_gop.png", dpi=120)
    plt.show()
    print("  Figure sauvegardee : output/part5_ratio_vs_gop.png")


#  5b -- Visualisation complete du pipeline

def visualize_pipeline(frames_bgr, encoded_video, decoded_video, quant_factor=10):

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Visualisation complete du pipeline MPEG-4 ", fontsize=16, fontweight='bold')
    gs  = gridspec.GridSpec(5, 4, figure=fig, hspace=0.5, wspace=0.35)

    #  frames originales
    for idx in range(min(4, len(frames_bgr))):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2RGB))
        ftype = encoded_video[idx]['type']
        ax.set_title(f"Frame {idx} ({ftype})", fontsize=9)
        ax.axis('off')

    # Canaux Y, Cb, Cr de la frame 0
    data = preprocess(frames_bgr[0])
    for ax, img, title in [
        (fig.add_subplot(gs[1, 0]), cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB), "Original"),
        (fig.add_subplot(gs[1, 1]), data['Y'],  "Canal Y (Luminance)"),
        (fig.add_subplot(gs[1, 2]), data['Cb'], "Canal Cb"),
        (fig.add_subplot(gs[1, 3]), data['Cr'], "Canal Cr"),
    ]:
        kw = {} if title == "Original" else {'cmap': 'gray'}
        ax.imshow(img, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # Pipeline bloc 8x8 DCT 
    Y    = data['Y']
    H, W = Y.shape
    Q    = get_quant_matrix(quant_factor)
    y0   = (H // 2 // 8) * 8
    x0   = (W // 2 // 8) * 8
    bloc_raw = Y[y0:y0+8, x0:x0+8].astype(np.float32)
    bloc_dct = cv2.dct(bloc_raw)
    bloc_q   = np.round(bloc_dct / Q).astype(np.int16)
    bloc_rec = np.clip(cv2.idct(bloc_q.astype(np.float32) * Q), 0, 255).astype(np.uint8)

    for ax, img, title, kw in [
        (fig.add_subplot(gs[2, 0]), bloc_raw, "Bloc 8x8 original",     {'cmap':'gray','vmin':0,'vmax':255}),
        (fig.add_subplot(gs[2, 1]), np.log1p(np.abs(bloc_dct)), "Apres DCT", {'cmap':'hot'}),
        (fig.add_subplot(gs[2, 2]), bloc_q,   "Apres Quantification",  {'cmap':'RdBu','vmin':-20,'vmax':20}),
        (fig.add_subplot(gs[2, 3]), bloc_rec, "Bloc reconstruit",      {'cmap':'gray','vmin':0,'vmax':255}),
    ]:
        ax.imshow(img, **kw); ax.set_title(title, fontsize=9); ax.axis('off')

    # Vecteurs de mouvement sur P-frame 
    pframe_idx = next((i for i, e in enumerate(encoded_video) if e['type'] == 'P'), 1)
    enc_p      = encoded_video[pframe_idx]
    mv         = enc_p['motion_vectors']
    orig_p_rgb = cv2.cvtColor(frames_bgr[pframe_idx], cv2.COLOR_BGR2RGB)

    ax = fig.add_subplot(gs[3, 0:2])
    ax.imshow(orig_p_rgb)
    nb_H, nb_W = mv.shape[:2]
    for i in range(nb_H):
        for j in range(nb_W):
            dy, dx = mv[i, j]
            y = i * MACRO_SIZE + MACRO_SIZE // 2
            x = j * MACRO_SIZE + MACRO_SIZE // 2
            if abs(dy) > 1 or abs(dx) > 1:
                ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                             arrowprops=dict(arrowstyle='->', color='red', lw=0.7))
    ax.set_title(f"Vecteurs de mouvement | Frame {pframe_idx}", fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[3, 2:4])
    ax.imshow(cv2.cvtColor(decoded_video[pframe_idx], cv2.COLOR_BGR2RGB))
    ax.set_title(f"P-frame {pframe_idx} reconstruite", fontsize=9)
    ax.axis('off')

    # Residuel 
    decoded_residual = decode_iframe(enc_p['residual']).astype(np.float32)
    spatial_residual = decoded_residual - 128.0
    ax.imshow(spatial_residual, cmap='RdBu', vmin=-60, vmax=60)

    ax = fig.add_subplot(gs[4, 0:2])
    ax.imshow(spatial_residual, cmap='gray')
    ax.set_title(f"Residual Map | Frame {pframe_idx}", fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[4, 2:4])
    ax.imshow(cv2.cvtColor(decoded_video[0], cv2.COLOR_BGR2RGB))
    ax.set_title("I-frame 0 reconstruite", fontsize=9)
    ax.axis('off')

    plt.savefig("output/part5_pipeline_complet.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  Figure sauvegardee : output/part5_pipeline_complet.png")