"""Entrena y evalúa una escena 3DGS en un proceso propio.

Envuelve `train.training` del repo oficial (third_party/gaussian-splatting) sin modificarlo y añade
lo que el repo no registra:

- Reanudación: usa el último `chkpntN.pth` y omite el entrenamiento si ya terminó.
- Registro continuo: `train_log.csv` (pérdida, nº de gaussianas, VRAM) y `gpu_log_*.csv` (temperatura,
  potencia, memoria real según nvidia-smi).
- Evaluación contra las imágenes ORIGINALES (`--gt_images`). `metrics.py` compara con el GT de la
  carpeta de entrada, que en una escena SPC ya está degradado.
- Estado en `run_status.json` y resumen en `metrics.json`, escritos al terminar cada etapa.

Códigos de salida: 0 = ok, 3 = sin VRAM (OOM), 1 = otro error.
Lo invoca `02_gaussian_splatting.ipynb`; también puede usarse por línea de comandos.
"""
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GS_REPO = REPO_ROOT / "third_party" / "gaussian-splatting"
sys.path.insert(0, str(GS_REPO))

import numpy as np
import torch
from PIL import Image

import train as gs_train  # train.py del repo oficial; su bloque __main__ no se ejecuta al importar
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import GaussianModel, Scene
from utils.general_utils import PILtoTorch, safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim

GPU_QUERY = "timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.gr"
GPU_HEADER = "timestamp,temp_c,power_w,util_pct,mem_mib,clock_mhz\n"
EXIT_OOM = 3


def write_json(path, payload):
    """Escribe en un archivo temporal y renombra, para no dejar JSON truncados si se corta la luz."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def start_gpu_logger(model_path, interval_s):
    n = len(list(model_path.glob("gpu_log_*.csv")))
    path = model_path / f"gpu_log_{n:02d}.csv"
    handle = open(path, "w", encoding="utf-8", newline="")
    handle.write(GPU_HEADER)
    handle.flush()
    proc = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits", "-l", str(interval_s)],
        stdout=handle, stderr=subprocess.DEVNULL,
    )
    return proc, handle


def summarize_gpu_logs(model_path):
    temps, mems, powers = [], [], []
    for path in sorted(model_path.glob("gpu_log_*.csv")):
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle, skipinitialspace=True):
                try:
                    temps.append(float(row["temp_c"]))
                    mems.append(float(row["mem_mib"]))
                    powers.append(float(row["power_w"]))
                except (TypeError, ValueError):
                    continue  # última línea cortada al terminar nvidia-smi, o "[N/A]"
    if not temps:
        return {}
    return {
        "gpu_temp_max_c": max(temps),
        "gpu_temp_mean_c": sum(temps) / len(temps),
        "gpu_mem_peak_mib": max(mems),
        "gpu_power_mean_w": sum(powers) / len(powers),
        "gpu_samples": len(temps),
    }


def latest_checkpoint(model_path):
    found = []
    for path in model_path.glob("chkpnt*.pth"):
        match = re.fullmatch(r"chkpnt(\d+)\.pth", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return max(found)[1] if found else None


def install_train_logger(model_path, every, resume_iter):
    """Envuelve `training_report` para registrar la curva de entrenamiento sin tocar el repo.

    Al reanudar descarta las filas posteriores al checkpoint y continúa su tiempo acumulado, de modo
    que el CSV queda sin iteraciones duplicadas. Devuelve (archivo, función de tiempo acumulado).
    """
    log_path = model_path / "train_log.csv"
    header = ["iteration", "elapsed_s", "loss", "l1", "n_gaussians", "vram_alloc_mib", "vram_peak_mib"]
    kept = []
    if log_path.exists():
        with open(log_path, encoding="utf-8", newline="") as handle:
            kept = [row for row in csv.reader(handle) if row and row[0] != "iteration" and int(row[0]) <= resume_iter]
    offset = float(kept[-1][1]) if kept else 0.0
    handle = open(log_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(header)
    writer.writerows(kept)
    handle.flush()
    original = gs_train.training_report
    start = time.perf_counter()

    def clock():
        return offset + time.perf_counter() - start

    def report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, render_func, render_args):
        if iteration % every == 0:
            writer.writerow([
                iteration, round(clock(), 2), loss.item(), Ll1.item(),
                scene.gaussians.get_xyz.shape[0],
                round(torch.cuda.memory_allocated() / 2**20, 1),
                round(torch.cuda.max_memory_allocated() / 2**20, 1),
            ])
            handle.flush()
        original(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, render_func, render_args)

    gs_train.training_report = report
    return handle, clock


def ply_vertex_count(path):
    with open(path, "rb") as handle:
        while True:
            line = handle.readline()
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.strip() == b"end_header" or not line:
                raise RuntimeError(f"PLY sin 'element vertex': {path}")


def find_gt_image(gt_dir, image_name):
    matches = [p for p in gt_dir.iterdir() if p.stem == image_name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Se esperaba una imagen original '{image_name}.*' en {gt_dir}; hay {len(matches)}.")
    return matches[0]


@torch.no_grad()
def evaluate(args, dataset, pipe, model_path, gt_dir, iterations):
    """PSNR/SSIM/LPIPS en test con las funciones del repo (las mismas del paper, §7.2)."""
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    lpips_net = LPIPS("vgg").cuda().eval()
    dataset.data_device = "cpu"  # las vistas de entrenamiento no se usan aquí
    gaussians = GaussianModel(dataset.sh_degree)
    scene = None
    results = {}
    for iteration in iterations:
        ply = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if scene is None:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        else:
            gaussians.load_ply(str(ply))
        cams = scene.getTestCameras()
        if not cams:
            raise RuntimeError("No hay vistas de test: falta --eval o hay muy pocas vistas.")
        out_dir = model_path / "test_renders" / f"iteration_{iteration}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        render_time = 0.0
        for cam in cams:
            torch.cuda.synchronize()
            start = time.perf_counter()
            image = torch.clamp(render(cam, gaussians, pipe, background)["render"], 0.0, 1.0)
            torch.cuda.synchronize()
            render_time += time.perf_counter() - start
            pil_gt = Image.open(find_gt_image(gt_dir, cam.image_name)).convert("RGB")
            gt_orig = PILtoTorch(pil_gt, (cam.image_width, cam.image_height))[:3].cuda()
            gt_input = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
            rows.append({
                "view": cam.image_name,
                "psnr": psnr(image[None], gt_orig[None]).item(),
                "ssim": ssim(image[None], gt_orig[None]).item(),
                "lpips": lpips_net(image[None], gt_orig[None]).item(),
                "psnr_vs_input": psnr(image[None], gt_input[None]).item(),
                "ssim_vs_input": ssim(image[None], gt_input[None]).item(),
                "lpips_vs_input": lpips_net(image[None], gt_input[None]).item(),
            })
            array = (image.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(array).save(out_dir / f"{cam.image_name}.png")
        with open(model_path / f"per_view_{iteration}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "view"}
        summary.update(
            n_test_views=len(rows),
            n_gaussians=ply_vertex_count(ply),
            ply_mb=ply.stat().st_size / 2**20,
            render_fps=len(rows) / render_time,
        )
        results[str(iteration)] = summary
        print(f"[EVAL {iteration}] " + " ".join(f"{k}={v:.4f}" for k, v in summary.items()), flush=True)
    return results


def main():
    parser = ArgumentParser(description="Entrena y evalúa 3DGS con registro de VRAM y temperatura")
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument("--gt_images", required=True, help="Carpeta con las imágenes originales de referencia")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--eval_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--gpu_log_interval", type=int, default=15)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    args.model_path = str(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_images).resolve()
    save_iters = sorted(set(i for i in args.save_iterations if i <= args.iterations) | {args.iterations})
    eval_iters = sorted(set(args.eval_iterations or save_iters))
    args.save_iterations = save_iters

    status_path = model_path / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.setdefault("attempts", [])
    attempt = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:]}
    status["attempts"].append(attempt)
    status["state"] = "running"
    write_json(status_path, status)

    gpu_proc, gpu_handle = start_gpu_logger(model_path, args.gpu_log_interval)
    exit_code = 0
    try:
        safe_state(args.quiet)
        # safe_state fija 0; aquí se aplica la semilla pedida con el mismo mecanismo.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        network_gui.try_connect = lambda: None  # sin visor: no abrir sockets durante corridas largas
        # Los checkpoints son propios y contienen tuplas; train.py no pasa weights_only.
        original_load = torch.load
        gs_train.torch.load = lambda path, **kw: original_load(path, **{"weights_only": False, **kw})

        dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
        final_ply = model_path / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
        if status.get("train_done") and final_ply.exists():
            print("Entrenamiento ya completo; solo se evalúa.", flush=True)
        else:
            checkpoint = latest_checkpoint(model_path)
            attempt["resumed_from"] = checkpoint.name if checkpoint else None
            resume_iter = int(re.findall(r"\d+", checkpoint.name)[0]) if checkpoint else 0
            log_handle, clock = install_train_logger(model_path, args.log_every, resume_iter)
            torch.cuda.reset_peak_memory_stats()
            gs_train.training(dataset, opt, pipe, args.test_iterations, save_iters, args.checkpoint_iterations,
                              str(checkpoint) if checkpoint else None, -1)
            torch.cuda.synchronize()
            log_handle.close()
            status["train_time_s"] = clock()
            status["torch_vram_peak_mib"] = max(status.get("torch_vram_peak_mib", 0.0),
                                                torch.cuda.max_memory_allocated() / 2**20)
            status["torch_vram_reserved_peak_mib"] = max(status.get("torch_vram_reserved_peak_mib", 0.0),
                                                         torch.cuda.max_memory_reserved() / 2**20)
            status["train_done"] = True
            write_json(status_path, status)

        torch.cuda.empty_cache()
        results = evaluate(args, dataset, pipe, model_path, gt_dir, eval_iters)
        summary = {
            "scene": args.source_path,
            "gt_images": str(gt_dir),
            "iterations": args.iterations,
            "seed": args.seed,
            "resolution_arg": args.resolution,
            "data_device": args.data_device,
            "densify_grad_threshold": args.densify_grad_threshold,
            "densify_until_iter": args.densify_until_iter,
            "train_time_s": status.get("train_time_s"),
            "torch_vram_peak_mib": status.get("torch_vram_peak_mib"),
            "torch_vram_reserved_peak_mib": status.get("torch_vram_reserved_peak_mib"),
            "eval": results,
        }
        write_json(model_path / "metrics.json", summary)
        status["state"] = "done"
    except torch.cuda.OutOfMemoryError as error:
        status["state"] = "oom"
        status["error"] = str(error)
        exit_code = EXIT_OOM
        print(f"\nOOM: la VRAM no alcanzó. {error}\nReanudable desde el último checkpoint.", file=sys.stderr, flush=True)
    except Exception as error:  # noqa: BLE001 - se registra y el notebook sigue con la corrida siguiente
        message = str(error)
        status["state"] = "oom" if "out of memory" in message.lower() else "failed"
        status["error"] = traceback.format_exc()
        exit_code = EXIT_OOM if status["state"] == "oom" else 1
        print(status["error"], file=sys.stderr, flush=True)
    finally:
        gpu_proc.terminate()
        gpu_proc.wait(timeout=30)
        gpu_handle.close()
        attempt["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        attempt["state"] = status["state"]
        status.update(summarize_gpu_logs(model_path))
        write_json(status_path, status)
        metrics_path = model_path / "metrics.json"
        if metrics_path.exists() and status["state"] == "done":
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.update({k: v for k, v in status.items() if k.startswith("gpu_")})
            write_json(metrics_path, metrics)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
