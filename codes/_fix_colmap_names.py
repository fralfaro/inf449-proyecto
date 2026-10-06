import json
import re
from pathlib import Path

p = Path('proyecto-final/notebook.ipynb')
d = json.loads(p.read_text())
cells = d['cells']

replacement = r'''def filter_sparse_to_available_images(scene_folder):
    """Alinea images/ con COLMAP usando el stem y filtra vistas sin pose."""
    sparse = scene_folder / 'sparse' / '0'
    images_dir = scene_folder / 'images'
    available_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    binary = sparse / 'images.bin'
    if not binary.exists():
        raise FileNotFoundError(
            f'No hay images.bin en {sparse}; no puedo filtrar las poses automáticamente.'
        )

    records = read_colmap_images_binary(binary)

    def pose_key(name):
        # Hace coincidir 000001.jpg, 000001.png y 000001 si representan la misma vista.
        return Path(name).stem.lower()

    pose_by_key = {}
    for record in records:
        key = pose_key(record[4])
        if key in pose_by_key:
            raise RuntimeError(f'Hay poses COLMAP duplicadas para la clave {key!r}.')
        pose_by_key[key] = record

    matched = []
    unmatched = []
    for image_path in available_paths:
        record = pose_by_key.get(pose_key(image_path.name))
        if record is None:
            unmatched.append(image_path)
        else:
            matched.append((image_path, record))

    if not matched:
        raise RuntimeError(
            f'{scene_folder.name}: ninguna de las {len(available_paths)} imágenes '
            'seleccionadas tiene una pose COLMAP compatible.'
        )

    # Una imagen sin pose no puede entrar a train.py. Se elimina solo de la copia
    # generada en RUN_ROOT, nunca del dataset original.
    for image_path in unmatched:
        image_path.unlink()
    if unmatched:
        print(
            f'[{scene_folder.name}] ADVERTENCIA: se descartaron {len(unmatched)} '
            f'vistas sin pose COLMAP: {[path.name for path in unmatched[:5]]}'
        )

    rename_pairs = []
    selected_records = []
    for image_path, record in matched:
        pose_name = Path(record[4]).name
        normalized_record = (
            record[0], record[1], record[2], record[3], pose_name, record[5], record[6]
        )
        selected_records.append(normalized_record)
        destination = images_dir / pose_name
        if image_path.name != pose_name:
            if destination.exists() and destination.resolve() != image_path.resolve():
                raise RuntimeError(
                    f'Colisión al alinear {image_path.name} con la pose {pose_name}.'
                )
            rename_pairs.append((image_path, destination))

    # Renombrado en dos fases para evitar colisiones entre nombres cruzados.
    temporary_pairs = []
    for index, (source, destination) in enumerate(rename_pairs):
        temporary = images_dir / f'.pose_tmp_{index}_{source.name}'
        source.rename(temporary)
        temporary_pairs.append((temporary, destination))
    for temporary, destination in temporary_pairs:
        temporary.rename(destination)

    write_colmap_images_binary(binary, selected_records)

    # El evaluador usa este manifiesto; actualizamos nombres y quitamos descartadas.
    manifest_path = scene_folder / 'image_manifest.csv'
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        dropped_names = {path.name for path in unmatched}
        rename_map = {source.name: destination.name for source, destination in rename_pairs}
        entry_names = manifest['entrada_3d'].map(lambda value: Path(str(value)).name)
        manifest = manifest[~entry_names.isin(dropped_names)].copy()
        manifest['image_name'] = entry_names[manifest.index].map(
            lambda name: rename_map.get(name, name)
        )
        manifest['entrada_3d'] = manifest['image_name'].map(
            lambda name: str(images_dir / name)
        )
        manifest.to_csv(manifest_path, index=False)

    return {
        'poses': len(selected_records),
        'imagenes': len(selected_records),
        'descartadas_sin_pose': len(unmatched),
        'renombradas_para_colmap': len(rename_pairs),
    }
'''

for cell in cells:
    text = ''.join(cell.get('source', []))
    if 'def filter_sparse_to_available_images(scene_folder):' in text:
        pattern = r'def filter_sparse_to_available_images\(scene_folder\):.*?\ndef prepare_one_scene'
        updated = re.sub(pattern, replacement + '\n\ndef prepare_one_scene', text, count=1, flags=re.S)
        if updated == text:
            raise RuntimeError('No pude reemplazar filter_sparse_to_available_images.')
        cell['source'] = updated.splitlines(keepends=True)
        cell['outputs'] = []
        cell['execution_count'] = None
        break
else:
    raise RuntimeError('No encontré la celda de entrenamiento 3DGS.')

p.write_text(json.dumps(d, ensure_ascii=False, indent=1) + '\n')
