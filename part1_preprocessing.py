
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#

def load_frames(chemin_frames: str) -> list[np.ndarray]:

    extensions = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted([
        f for f in os.listdir(chemin_frames)
        if f.lower().endswith(extensions)
    ])

    if not files:
        raise FileNotFoundError(
            f"Aucune image trouvée dans le dossier : {chemin_frames}"
        )


    WIDTH, HEIGHT = 640, 360

    frames = []
    for fname in files:
        path = os.path.join(chemin_frames, fname)
        img = cv2.imread(path)          
        if img is None:
            print(f"  [AVERTISSEMENT] Impossible de lire : {fname}")
            continue
        h, w = img.shape[:2]
        if w > WIDTH or h > HEIGHT:
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(img)

    print(f"  {len(frames)} frame(s) chargee(s) depuis frames/ -> resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    return frames


"""
def load_frames(chemin_frames: str) -> list[np.ndarray]:

donc cette ligne est 
une fonction qui prend comme argument le chemin de dossier des frames 
et renvoie une liste contenant les matrices représentant les images 

extensions = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted([
        f for f in os.listdir(chemin_frames)
        if f.lower().endswith(extensions)
    ])
on définit les extensions possibles
puis on affecte à files les images ordonnées pour préserver l'odre des 
frames dans la vidéo . si files est vide on déclenche une exception 
 WIDTH, HEIGHT = 640, 360

    frames = []
    for fname in files:
        path = os.path.join(chemin_frames, fname)
        img = cv2.imread(path) 
on parcourt files, la liste des frames , on lit l'image en BGR si y a 
aucune on avertit et continue , sinon on récupère les dimensions de 
l'image et on les compare avec largeur et hauteur 640*360 si > on 
redimensionne en préservant la qualité grace à cv2.INTER_AREA , 
et on stocke l'image résultante dans la liste frames 


2. Le fonctionnement du RedimensionnementTu as bien noté l'utilisation
 de cv2.INTER_AREA.Cette méthode est spécifiquement conçue pour le "downsampling"
(réduction de taille).  Au lieu d'ignorer des pixels, elle fait une moyenne locale 
 des couleurs, ce qui évite l'effet d'escalier (aliasing) sur les lignes fines de tes
 images originales.3. Structure de la matrice (np.ndarray)Quand tu dis que cela renvoie 
 des matrices :Chaque élément de ta liste frames est en fait un tableau à 3 dimensions. 
Les dimensions sont : (Hauteur, Largeur, Canaux de couleur).Pour une image redimensionnée,
 la structure exacte sera (360, 640, 3).
"""


# ─────────────────────────────────────────────
#  Conversion BGR → YCbCr
# ─────────────────────────────────────────────
"""
la fonction prend en entrée la liste des images, convertie chacune en YCrCb 
puis découpe chaque image en 3 images chacune portant une des canaux Y , Cr et Cb
 et renvoie à la fin une liste de  ces triplets 
"""
def bgr_to_ycbcr(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

  
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

    Y, Cr, Cb = cv2.split(ycrcb)   

    return Y, Cb, Cr


"""
on prend les matrices Cb et Cr résultantes de la convertion vers YCrCB
avec  Cb[::2, ::2] on prend une ligne sur 2 et 1 colonne sur 2
donc on supprime les couleurs  mais on garde la luminance Y vu qu'elle est plus 
remarquée par l'oeil humain 
sous-échantillonnage 4:2:0
"""

def sous_échantillonnage(Cb: np.ndarray, Cr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]

    return Cb_sub, Cr_sub

"""
cette fonction englobe le travail de la partie 1 : pre-processing 
elle récupère les dimensions de l'image avant de commencer les changements
elle fait appel à toutes les fonctions précédentes et garde les résultat 
dans un dictionnaire ( clé, valeur ) pour les utiliser dans les prochaines parties 
"""

def preprocess(frame_bgr: np.ndarray) -> dict:

    H, W = frame_bgr.shape[:2]

    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)

    Cb_sub, Cr_sub = sous_échantillonnage(Cb, Cr)

    return {
        "Y":      Y,
        "Cb":     Cb,
        "Cr":     Cr,
        "Cb_sub": Cb_sub,
        "Cr_sub": Cr_sub,
        "shape":  (H, W),
    }

"""
cette fonction reconstruit l'image original 
elle récupère les informations d'après le dictionnaire puis 
reconstruit les canaux Cb et Cr , 
INTER_LINEAR crée un flou doux pour lisser les pixels manquants.
à la fin elle fusionne les 3 images Y, Cb et Cr et converti vers BGR 
"""

def reconstruct_bgr(preprocessed: dict) -> np.ndarray:

    H, W    = preprocessed["shape"]
    Y       = preprocessed["Y"]
    Cb_sub  = preprocessed["Cb_sub"]
    Cr_sub  = preprocessed["Cr_sub"]

    Cb_up = cv2.resize(Cb_sub, (W, H), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_sub, (W, H), interpolation=cv2.INTER_LINEAR)

    ycrcb = cv2.merge([Y, Cr_up, Cb_up])

    frame_bgr_reconstructed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return frame_bgr_reconstructed




def visualize_preprocessing(frame_bgr: np.ndarray, frame_index: int = 0):

    data = preprocess(frame_bgr)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle(f"Partie 1 — Pré-traitement | Frame {frame_index}", fontsize=13)

    axes[0].imshow(frame_rgb)
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(data["Y"], cmap="gray")
    axes[1].set_title("Canal Y (Luminance)")
    axes[1].axis("off")

    axes[2].imshow(data["Cb"], cmap="gray")
    axes[2].set_title("Canal Cb (pleine résol.)")
    axes[2].axis("off")

    axes[3].imshow(data["Cr"], cmap="gray")
    axes[3].set_title("Canal Cr (pleine résol.)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(f"output/part1_visualization_frame{frame_index:03d}.png", dpi=120)
    plt.show()
    print(f"  Figure sauvegardée dans output/")

