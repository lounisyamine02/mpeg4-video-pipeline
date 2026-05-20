

import zlib
import pickle
import os
import numpy as np

"""
dans cette partie , on fait une compression sans
 perte pour optimiser l'espace  pris par les matrices 
 et tout enregistrer dans un fichier.bin
"""

def entropy_encode(encoded_video: list, output_path: str = "output/video_compressed.bin"):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  Serialisation des donnees")

    # Etape 1 : Serialisation avec pickle
    raw_bytes = pickle.dumps(encoded_video)
    raw_size  = len(raw_bytes)
    print(f"  Taille apres serialisation : {raw_size:,} octets")

    # Etape 2 : Compression zlib 
    print(f"  Compression zlib en cours")
    compressed_bytes = zlib.compress(raw_bytes, level=6)
    compressed_size  = len(compressed_bytes)
    print(f"  Taille apres compression   : {compressed_size:,} octets")

    # Etape 3 : Ecriture dans le fichier .bin
    with open(output_path, 'wb') as f:
        f.write(compressed_bytes)
    print(f"  Fichier sauvegarde : {output_path}")

    # Calcul du ratio
    ratio = raw_size / compressed_size

    stats = {
        'raw_size':        raw_size,
        'compressed_size': compressed_size,
        'ratio':           ratio,
        'output_path':     output_path,
    }

    return stats



def entropy_decode(bin_path: str = "output/video_compressed.bin") -> list:

    print(f"  Lecture du fichier : {bin_path}")

    # Etape 1 : Lecture
    with open(bin_path, 'rb') as f:
        compressed_bytes = f.read()
    print(f"  Taille lue : {len(compressed_bytes):,} octets")

    # Etape 2 : Decompression zlib
    print(f"  Decompression zlib...")
    raw_bytes = zlib.decompress(compressed_bytes)
    print(f"  Taille decompresse : {len(raw_bytes):,} octets")

    # Etape 3 : Deserialisation pickle
    encoded_video = pickle.loads(raw_bytes)
    print(f"  {len(encoded_video)} frames recuperees")

    return encoded_video


#  Calcul taille originale

def compute_original_size(frames_bgr: list) -> int:

    total = 0
    for frame in frames_bgr:
        H, W = frame.shape[:2]
        total += H * W * 3
    return total

