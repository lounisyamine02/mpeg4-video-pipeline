import os, sys, cv2
from part1_preprocessing import load_frames, preprocess, visualize_preprocessing
from part2_iframe        import encode_all_iframe, decode_all_iframe, visualize_iframe
from part3_pframe        import encode_video, decode_pframe, visualize_pframe
from part4_entropy       import entropy_encode, entropy_decode, compute_original_size
from part5_evaluation    import compute_metrics, compare_ratio_quant, compare_ratio_gop, visualize_pipeline

FRAMES_DIR   = "frames"
OUTPUT_DIR   = "output"
QUANT_FACTOR = 10

def main():
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else FRAMES_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 30)
    print(" Multimedia Mini Project : Simplified MPEG-4 Video Encoder Pipeline")
    print("=" * 30)

    # PARTIE 1
    print("\n[PARTIE 1] Chargement et pre-traitement")
    frames_bgr = load_frames(frames_dir)
    preprocessed_frames = [preprocess(f) for f in frames_bgr]
    print(f" {len(preprocessed_frames)} frames pre-traitees")
    visualize_preprocessing(frames_bgr[0], frame_index=0)

    # PARTIE 2
    print("\n[PARTIE 2] Encodage I-frames")
    enc0      = encode_all_iframe(preprocessed_frames[0], quant_factor=QUANT_FACTOR)
    dec0      = decode_all_iframe(enc0)
    orig_size = frames_bgr[0].shape[0] * frames_bgr[0].shape[1] * 3
    comp_size = enc0['Y']['coeffs'].nbytes + enc0['Cb']['coeffs'].nbytes + enc0['Cr']['coeffs'].nbytes
    print(f"  Ratio compression : {orig_size / comp_size:.2f}x")
    cv2.imwrite(f"{OUTPUT_DIR}/part2_decoded_frame000.png", dec0)
    visualize_iframe(frames_bgr[0], quant_factor=QUANT_FACTOR, frame_index=0)

    # PARTIE 3
    print("\n[PARTIE 3] Encodage P-frames")
    encoded_video, decoded_video = encode_video(
        preprocessed_frames, quant_factor=QUANT_FACTOR, gop_size=5
    )
    i_count = sum(1 for e in encoded_video if e['type'] == 'I')
    p_count = sum(1 for e in encoded_video if e['type'] == 'P')
    print(f"  I-frames : {i_count} | P-frames : {p_count}")
    visualize_pframe(frames_bgr, encoded_video, decoded_video, frame_index=1)

    # PARTIE 4
    print("\n[PARTIE 4] Compression entropique -> .bin ")
    stats         = entropy_encode(encoded_video, f"{OUTPUT_DIR}/video_compressed.bin")
    bin_size      = os.path.getsize(f"{OUTPUT_DIR}/video_compressed.bin")
    original_size = compute_original_size(frames_bgr)
    print(f"  Ratio FINAL : {original_size / bin_size:.2f}x")

    #  decodage complet depuis .bin -> frames PNG
    print("\n[VERIFICATION] Decodage complet depuis le fichier .bin")
    recovered_encoded = entropy_decode(f"{OUTPUT_DIR}/video_compressed.bin")
    os.makedirs(f"{OUTPUT_DIR}/decoded_frames", exist_ok=True)

    ref_frame = None
    for i, enc in enumerate(recovered_encoded):
        if enc['type'] == 'I':
            frame = decode_all_iframe(enc)
        else:
            frame = decode_pframe(enc, ref_frame)
        ref_frame = frame
        cv2.imwrite(f"{OUTPUT_DIR}/decoded_frames/frame_{i:03d}.png", frame)

    print(f"  {len(recovered_encoded)} frames reconstruites dans output/decoded_frames/")

    # PARTIE 5
    print("\n[PARTIE 5] Evaluation & Visualisation")
    compute_metrics(frames_bgr, encoded_video, f"{OUTPUT_DIR}/video_compressed.bin")
    print("\n  Graphique ratio vs Fq")
    compare_ratio_quant(preprocessed_frames)
    print("\n  Graphique ratio vs GOP")
    compare_ratio_gop(preprocessed_frames)
    print("\n  Visualisation pipeline complet")
    visualize_pipeline(frames_bgr, encoded_video, decoded_video, quant_factor=QUANT_FACTOR)

    print("=" * 50)

if __name__ == "__main__":
    main()