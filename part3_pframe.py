
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from part2_iframe import encode_iframe, decode_iframe, encode_all_iframe, decode_all_iframe



MACRO_SIZE  = 16   # taille des macroblocs (16x16)
SEARCH_WIN  = 5    # fenetre de recherche +/- 5 pixels
GOP_SIZE    = 5    # toutes les 5 frames -> I-frame


"""
entrée la matrice de Y actuelle et sa précédente 
"""

def motion_estimation(current_Y: np.ndarray, reference_Y: np.ndarray, macro_size: int = MACRO_SIZE, search_win: int = SEARCH_WIN) -> np.ndarray:

    H, W = current_Y.shape

    # Nombre de macroblocs en hauteur et largeur
    nb_H = H // macro_size
    nb_W = W // macro_size

    # matrice des vecteurs de mouvement
    motion_vectors = np.zeros((nb_H, nb_W, 2), dtype=np.int16)

    for i in range(nb_H):
        for j in range(nb_W):

            # Coordonnees du macrobloc courant vers coordonnée de l'image 
            y = i * macro_size
            x = j * macro_size

            current_block = current_Y[y:y+macro_size, x:x+macro_size].astype(np.float32)
            #initialiser avant de chercher le meilleur déplacement 
            best_dy   = 0
            best_dx   = 0
            best_cost = float('inf')

            # Recherche dans la fenetre +/- search_win
            for dy in range(-search_win, search_win + 1):
                for dx in range(-search_win, search_win + 1):

                    # Coordonnees dans la frame reference
                    ry = y + dy
                    rx = x + dx
                    if ry < 0 or rx < 0 or ry + macro_size > H or rx + macro_size > W:
                        continue

                    ref_block = reference_Y[ry:ry+macro_size, rx:rx+macro_size].astype(np.float32)
                    cost = np.sum(np.abs(current_block - ref_block))

                    if cost < best_cost:
                        best_cost = cost
                        best_dy   = dy
                        best_dx   = dx

            motion_vectors[i, j] = [best_dy, best_dx]

    return motion_vectors


#  Calcul du Residuel

def calcul_residual(current_Y: np.ndarray, reference_Y: np.ndarray, motion_vectors: np.ndarray, macro_size: int = MACRO_SIZE) -> np.ndarray:

    H, W = current_Y.shape
    nb_H = H // macro_size
    nb_W = W // macro_size

    prediction = np.zeros((H, W), dtype=np.float32)

    for i in range(nb_H):
        for j in range(nb_W):
            y  = i * macro_size
            x  = j * macro_size
            dy, dx = motion_vectors[i, j]

            ry = y + dy
            rx = x + dx

            if ry < 0 or rx < 0 or ry + macro_size > H or rx + macro_size > W:
                prediction[y:y+macro_size, x:x+macro_size] = 128.0
            else:
                prediction[y:y+macro_size, x:x+macro_size] = \
                    reference_Y[ry:ry+macro_size, rx:rx+macro_size].astype(np.float32)

    # Residuel = difference entre frame courante et prediction
    current_f  = current_Y.astype(np.float32)
    residual   = current_f - prediction

    return residual


#  Encodeur P-frame 

def encode_pframe(preprocessed: dict, reference_decoded: np.ndarray, quant_factor: int = 10) -> dict:

    current_Y = preprocessed['Y']
    H, W      = preprocessed['shape']

    # Extraire le canal Y de la frame reference 
    ref_ycrcb   = cv2.cvtColor(reference_decoded, cv2.COLOR_BGR2YCrCb)
    reference_Y = ref_ycrcb[:, :, 0]

    if reference_Y.shape != current_Y.shape:
        reference_Y = cv2.resize(reference_Y, (W, H))

    # Etape 1 : Motion estimation
    motion_vectors = motion_estimation(current_Y, reference_Y)

    # Etape 2 : Calcul du residuel
    residual = calcul_residual(current_Y, reference_Y, motion_vectors)

    # Etape 3 : DCT + quantification du residuel
    residual_shifted = np.clip(residual + 128, 0, 255).astype(np.uint8)
    residual_encoded = encode_iframe(residual_shifted, quant_factor)

    # Etape 4 : Cb et Cr encodés directement (comme I-frame)
    Cb_encoded = encode_iframe(preprocessed['Cb_sub'], quant_factor)
    Cr_encoded = encode_iframe(preprocessed['Cr_sub'], quant_factor)

    return {
        'type':           'P',
        'motion_vectors': motion_vectors,
        'residual':       residual_encoded,
        'Cb':             Cb_encoded,
        'Cr':             Cr_encoded,
        'original_shape': (H, W),
        'quant_factor':   quant_factor,
    }


#  Decodeur P-frame 

def decode_pframe(encoded_pframe: dict, reference_decoded: np.ndarray) -> np.ndarray:

    H, W           = encoded_pframe['original_shape']
    motion_vectors = encoded_pframe['motion_vectors']
    quant_factor   = encoded_pframe['quant_factor']

    # Extraire Y de la reference
    ref_ycrcb   = cv2.cvtColor(reference_decoded, cv2.COLOR_BGR2YCrCb)
    reference_Y = ref_ycrcb[:, :, 0]
    if reference_Y.shape != (H, W):
        reference_Y = cv2.resize(reference_Y, (W, H))

    # Etape 1 : Decoder le residuel
    residual_shifted = decode_iframe(encoded_pframe['residual']).astype(np.float32)
    residual         = residual_shifted - 128.0

    # Etape 2 : Reconstruire la prediction
    macro_size  = MACRO_SIZE
    nb_H        = H // macro_size
    nb_W        = W // macro_size
    prediction  = np.zeros((H, W), dtype=np.float32)

    for i in range(nb_H):
        for j in range(nb_W):
            y       = i * macro_size
            x       = j * macro_size
            dy, dx  = motion_vectors[i, j]
            ry      = y + dy
            rx      = x + dx

            if ry < 0 or rx < 0 or ry + macro_size > H or rx + macro_size > W:
                prediction[y:y+macro_size, x:x+macro_size] = 128.0
            else:
                prediction[y:y+macro_size, x:x+macro_size] = \
                    reference_Y[ry:ry+macro_size, rx:rx+macro_size].astype(np.float32)

    # Etape 3 : Reconstruire Y = Prediction + Residuel
    Y_reconstructed = np.clip(prediction + residual, 0, 255).astype(np.uint8)

    # Etape 4 : Decoder Cb et Cr
    Cb = decode_iframe(encoded_pframe['Cb'])
    Cr = decode_iframe(encoded_pframe['Cr'])

    Cb_up = cv2.resize(Cb, (W, H), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr, (W, H), interpolation=cv2.INTER_LINEAR)

    # Recomposer YCrCb -> BGR
    ycrcb = cv2.merge([Y_reconstructed, Cr_up, Cb_up])
    bgr   = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return bgr


#  Pipeline complet GOP (I + P frames)

def encode_video(preprocessed_frames: list,
                 quant_factor: int = 10,
                 gop_size: int = GOP_SIZE) -> list:

    encoded_video = []
    decoded_video = []

    for i, preprocessed in enumerate(preprocessed_frames):

        if i % gop_size == 0:
            # ── I-frame ──
            print(f"  Frame {i:03d} -> I-frame")
            encoded = encode_all_iframe(preprocessed, quant_factor)
            encoded['type'] = 'I'
            decoded = decode_all_iframe(encoded)

        else:
            # ── P-frame ──
            print(f"  Frame {i:03d} -> P-frame")
            reference = decoded_video[-1]   
            encoded   = encode_pframe(preprocessed, reference, quant_factor)
            decoded   = decode_pframe(encoded, reference)

        encoded_video.append(encoded)
        decoded_video.append(decoded)

    return encoded_video, decoded_video


#  Visualisation

def visualize_pframe(frames_bgr: list, encoded_video: list, decoded_video: list, frame_index: int = 1):

    if encoded_video[frame_index]['type'] != 'P':
        frame_index += 1

    enc         = encoded_video[frame_index]
    orig_bgr    = frames_bgr[frame_index]
    ref_bgr     = decoded_video[frame_index - 1]
    recon_bgr   = decoded_video[frame_index]
    mv          = enc['motion_vectors']

    orig_rgb    = cv2.cvtColor(orig_bgr,  cv2.COLOR_BGR2RGB)
    ref_rgb     = cv2.cvtColor(ref_bgr,   cv2.COLOR_BGR2RGB)
    recon_rgb   = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(14, 6))
    fig.suptitle(f"Partie 3 -- P-frame | Frame {frame_index}", fontsize=13)

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Frame originale")
    axes[0].axis('off')

    axes[1].imshow(ref_rgb)
    axes[1].set_title("Frame reference (precedente)")
    axes[1].axis('off')

    axes[2].imshow(orig_rgb)
    axes[2].set_title("Vecteurs de mouvement")
    H, W    = orig_bgr.shape[:2]
    nb_H, nb_W = mv.shape[:2]
    for i in range(nb_H):
        for j in range(nb_W):
            dy, dx = mv[i, j]
            y = i * MACRO_SIZE + MACRO_SIZE // 2
            x = j * MACRO_SIZE + MACRO_SIZE // 2
            if abs(dy) > 2 or abs(dx) > 2:
                axes[2].annotate('',
                    xy=(x + dx, y + dy),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8)
                )
    axes[2].axis('off')

    axes[3].imshow(recon_rgb)
    axes[3].set_title("Frame reconstruite")
    axes[3].axis('off')

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/part3_visualization_frame{frame_index:03d}.png", dpi=120)
    plt.show()
    print(f"  Figure sauvegardee dans output/")
