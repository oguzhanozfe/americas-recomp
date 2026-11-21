"""Verify nearest-neighbor texture sampling with perspective-correct fragment UVs."""
import argparse
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


SIZE = 32
WIDTH, HEIGHT = 4, 3
CLEAR = (-777.0,) * 4
VERTEX = """#version 450
layout(location=0) out vec4 attr;
void main() {
    const vec2 ndc[3] = vec2[](vec2(-0.75,-0.75), vec2(0.75,-0.75), vec2(-0.75,0.25));
    const vec2 uv[3] = vec2[](vec2(-0.125,-0.0625), vec2(3.875,-0.0625), vec2(-0.125,5.9375));
    const float w[3] = float[](1,2,4);
    int i = gl_VertexIndex;
    gl_Position = vec4(ndc[i] * w[i], 0, w[i]);
    attr = vec4(uv[i], 0, 1);
}
"""


def fragment_program():
    load_texture = (0xF40C0202, 0xFA000000)
    load_sampler = (0xF4080402, 0xFA000020)
    wait = 0xBF8C0000
    words = [*load_texture, *load_sampler, wait]
    for channel in range(2):
        interp = 0xC8000000 | ((2 + channel) << 18) | (channel << 8)
        words.extend((interp, interp | (1 << 16) | 1))
    sample_lz = (0xF09C0F08, 0x00820402)
    export_rgba = (0xF800180F, 0x07060504)
    words.extend((*sample_lz, wait, *export_rgba, 0xBF810000))
    return struct.pack(f"<{len(words)}I", *words)


def color(x, y):
    return ((x + 1) / 4, (y + 1) / 4, (y * WIDTH + x + 1) / 16, 1.0)


def verify(image):
    if len(image) != SIZE * SIZE * 16:
        raise ValueError(f"unexpected RGBA32F byte count: {len(image)}")
    hits = set()
    covered = 0
    for i, rgba in enumerate(struct.iter_unpack("<4f", image)):
        y, x = divmod(i, SIZE)
        b = (x + 0.5 - 4) / 24
        c = (y + 0.5 - 4) / 16
        a = 1 - b - c
        expected = CLEAR
        if min(a, b, c) > 0:
            covered += 1
            reciprocal_w = a + b / 2 + c / 4
            u = -0.125 + 4 * (b / 2) / reciprocal_w
            v = -0.0625 + 6 * (c / 4) / reciprocal_w
            tx = max(0, min(WIDTH - 1, math.floor(u * WIDTH)))
            ty = max(0, min(HEIGHT - 1, math.floor(v * HEIGHT)))
            hits.add((tx, ty))
            expected = color(tx, ty)
        if rgba != expected:
            raise ValueError(f"pixel ({x}, {y}): expected {expected}, read {rgba}")
    if len(hits) != WIDTH * HEIGHT:
        raise ValueError(f"triangle exercised only {len(hits)} texture cells")
    print(f"PASS: {covered} textured, {SIZE * SIZE - covered} clear, {len(hits)} texels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec", type=Path,
                        default=Path(__file__).resolve().parents[2] / "build-xbox" / "exec.exe")
    runner = parser.parse_args().exec.resolve()
    compiler = shutil.which("glslangValidator")
    if not runner.is_file() or compiler is None:
        parser.error("provide the built --exec runner and add glslangValidator to PATH")
    with tempfile.TemporaryDirectory(prefix="texture triangle ") as temp:
        work = Path(temp)
        (work / "uv.vert").write_text(VERTEX, encoding="ascii")
        subprocess.run([compiler, "-V", "uv.vert", "-o", "uv.spv"], cwd=work, check=True, timeout=60)
        (work / "sample.bin").write_bytes(fragment_program())
        texels = (color(x, y) for y in range(HEIGHT) for x in range(WIDTH))
        (work / "texture.bin").write_bytes(b"".join(struct.pack("<4f", *c) for c in texels))
        (work / "scenario.txt").write_text(
            f"stage ps\nwave 32\ntarget {SIZE} {SIZE}\ninputaddr 0x2\ninterp 1 0\n"
            f"image 1 {WIDTH} {HEIGHT} init texture.bin\nregion 0 256\ntimg 0 0 1\npoke 0 32 0x92\nuser 4 0\n"
            "vs uv.spv\ncode sample.bin\nexport 0 9 0xe4\nout out\n", encoding="ascii")
        subprocess.run([runner, "scenario.txt"], cwd=work, check=True, timeout=90)
        verify((work / "out" / "target0.bin").read_bytes())


if __name__ == "__main__":
    main()
