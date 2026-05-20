
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# -------------------------------------------------
#  Matrice de quantification standard JPEG (8x8)
# -------------------------------------------------

"""
notre image sera découpée en bloc 8*8, chaque bloc sera diviser par 
quant_matrix_base (un bloc 8*8 ou la partie haut gauche contiendra 
les basses fréquences de l'image et en bas à droite les hautes fréquences.
basses fréquence ---> changement lent
hautes fréquence ---> changement rapide  ) 
 pour éliminer les détails non remarquables par l'oeil humains. 
 résultat un bloc rempli de 0 en bas à droite qui sera facile à 
 stocker après 


"""

QUANT_MATRIX_BASE = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)


def get_quant_matrix(quant_factor: int = 10) -> np.ndarray:

    Q = np.clip(QUANT_MATRIX_BASE * quant_factor / 10.0, 1, 255)
    return Q.astype(np.float32)


# -------------------------------------------------
#  Padding
# -------------------------------------------------
""" 
pour gérer le cas ou les dimensions de l'image des canaux n'est pas divisible par 8
en entrée une matrice représentant un canal et taille de bloc 8*8
on prend les dimensions de la maatrice 
on calcule le nombres de lignes à ajouter en bas et colonnes à droite 
remplir le padding avec 0 ( le noir) 
retourne l'image avec padding dans padded
"""
def pad_canal(canal: np.ndarray, block_size: int = 8) -> np.ndarray:

    H, W = canal.shape
    pad_h = (block_size - H % block_size) % block_size #le nombres de lignes à ajouter en bas 
    pad_w = (block_size - W % block_size) % block_size  #le nombres de colonnes à ajouter à droite 

  
    padded = np.pad(canal, ((0, pad_h), (0, pad_w)), mode='constant')
    return padded


"""
Encodeur I-frame
entrée est la matrice d'un canal et facteur de qualité 
"""


def encode_iframe(canal: np.ndarray, quant_factor: int = 10) -> dict:

    H_orig, W_orig = canal.shape  #récupère les dimensions 
    Q = get_quant_matrix(quant_factor)  #générer la matrice de quantification $Q$ 

    # padding
    padded = pad_canal(canal, block_size=8)   
    H, W   = padded.shape

    """
    préparer une matrice pour stocker les blocs compressés en 
    Utilisant des entiers signés sur 16 bits
    """
    coeffs = np.zeros((H, W), dtype=np.int16)

    # parcours bloc par bloc 
    for y in range(0, H, 8):
        for x in range(0, W, 8):
            # Extraire les bloc 8x8 et convertir en float32 pour la fonction de DCT
            bloc = padded[y:y+8, x:x+8].astype(np.float32)

            # appliquer la DCT sur le bloc ( convertir du pixel en fréquence)
            dct_bloc = cv2.dct(bloc)

            # Quantification (division par Q)
            quantized = np.round(dct_bloc / Q).astype(np.int16)
            #remettre le bloc compressé à sa place 
            coeffs[y:y+8, x:x+8] = quantized
    #dictionnaire de données retourné
    return {
        'coeffs':         coeffs,
        'original_shape': (H_orig, W_orig),
        'padded_shape':   (H, W),
        'quant_factor':   quant_factor,
    }


"""  
Decodeur I-frame

"""
def decode_iframe(encoded: dict) -> np.ndarray:

    #récupère les données d'après le dictionnaire 
    coeffs         = encoded['coeffs'].astype(np.float32)
    H_orig, W_orig = encoded['original_shape']
    H, W           = encoded['padded_shape']
    Q              = get_quant_matrix(encoded['quant_factor'])
    # préparer une matrice pour stocker la matrice reconstituée 
    reconstructed = np.zeros((H, W), dtype=np.float32)

    # Parcours bloc par bloc
    for y in range(0, H, 8):
        for x in range(0, W, 8):
            bloc_q = coeffs[y:y+8, x:x+8]

            # De-quantification ( multiplication par Q)
            dct_bloc = bloc_q * Q

            # IDCT ( dct inverse) 
            bloc_rec = cv2.idct(dct_bloc)

            reconstructed[y:y+8, x:x+8] = bloc_rec

    # enlever le padding
    reconstructed = reconstructed[:H_orig, :W_orig]

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    return reconstructed


#  Encode / Decode une frame complete (Y + Cb + Cr)

def encode_all_iframe(preprocessed: dict, quant_factor: int = 10) -> dict:

    return {
        'Y':              encode_iframe(preprocessed['Y'],      quant_factor),
        'Cb':             encode_iframe(preprocessed['Cb_sub'], quant_factor),
        'Cr':             encode_iframe(preprocessed['Cr_sub'], quant_factor),
        'original_shape': preprocessed['shape'],
        'quant_factor':   quant_factor,
    }


def decode_all_iframe(encoded_frame: dict) -> np.ndarray:
 
    H_orig, W_orig = encoded_frame['original_shape']

    # Decoder chaque canal
    Y  = decode_iframe(encoded_frame['Y'])
    Cb = decode_iframe(encoded_frame['Cb'])
    Cr = decode_iframe(encoded_frame['Cr'])

    # retourner en Cb et Cr originaux qui ont été sous-echantillonnes en Partie 1
    Cb_up = cv2.resize(Cb, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    # Recomposer YCrCb 
    ycrcb = cv2.merge([Y, Cr_up, Cb_up])

    # Retour en BGR
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return bgr


# -------------------------------------------------
#  Visualisation (Part 5b)
# -------------------------------------------------

def visualize_iframe(frame_bgr: np.ndarray, quant_factor: int = 10, frame_index: int = 0):

    from part1_preprocessing import preprocess

    # Pre-traitement
    data  = preprocess(frame_bgr)
    Y     = data['Y']   #Extrait uniquement le canal de luminance Y
    Q     = get_quant_matrix(quant_factor)

    # Prendre le centre de l'imageun comme bloc representatif  + coordonnées du départ
    H, W  = Y.shape
    y0    = (H // 2 // 8) * 8
    x0    = (W // 2 // 8) * 8
    bloc_raw = Y[y0:y0+8, x0:x0+8].astype(np.float32)

    # DCT
    bloc_dct  = cv2.dct(bloc_raw)

    # Quantification
    bloc_q    = np.round(bloc_dct / Q).astype(np.int16)

    # IDCT (reconstruction)
    bloc_rec  = cv2.idct(bloc_q.astype(np.float32) * Q)
    bloc_rec  = np.clip(bloc_rec, 0, 255).astype(np.uint8)

    # Encoder et decoder la frame complete 
    encoded   = encode_all_iframe(data, quant_factor)
    decoded   = decode_all_iframe(encoded)
    orig_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    dec_rgb   = cv2.cvtColor(decoded,   cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"Partie 2 -- I-frame | Frame {frame_index} | Fq={quant_factor}",
        fontsize=13
    )

    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(bloc_raw, cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Bloc 8x8 original (Y)")
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(np.log1p(np.abs(bloc_dct)), cmap='hot')
    ax2.set_title("Apres DCT (log)")
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(bloc_q, cmap='RdBu', vmin=-20, vmax=20)
    ax3.set_title("Apres Quantification")
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(bloc_rec, cmap='gray', vmin=0, vmax=255)
    ax4.set_title("Bloc reconstruit (IDCT)")
    ax4.axis('off')

    # Ligne du bas : frame complete originale vs reconstruite
    ax5 = fig.add_subplot(2, 2, 3)
    ax5.imshow(orig_rgb)
    ax5.set_title("Frame originale")
    ax5.axis('off')

    ax6 = fig.add_subplot(2, 2, 4)
    ax6.imshow(dec_rgb)
    ax6.set_title(f"Frame reconstruite (Fq={quant_factor})")
    ax6.axis('off')

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/part2_visualization_frame{frame_index:03d}_fq{quant_factor}.png", dpi=120)
    plt.show()
    print(f"  Figure sauvegardee dans output/")

