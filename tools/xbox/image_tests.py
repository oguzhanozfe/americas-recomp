"""Image execution tests: formats other than RGBA32F, 1D and 3D images, and samplers built from the guest S#."""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdna_asm
import exec_tests as T

FLOAT = {0.2: 0x3e4ccccd, 0.6: 0x3f19999a, 1.0: 0x3f800000, 0.5: 0x3f000000, 1.25: 0x3fa00000}
DIM_1D, DIM_2D, DIM_3D = 0, 1, 2


def _image(r, image_id, fmt, count):
    return struct.unpack('<%d%s' % (count, fmt), open(os.path.join(r['out'], f"image{image_id}.bin"), 'rb').read())


def _out_vsharp(p, sgpr, off):
    p.smem('s_load_dwordx4', sgpr, 4, off)


def test_store_rgba8_unorm():
    p = rdna_asm.Program()
    p.smem('s_load_dwordx8', 8, 4, 0)
    p.sopp('s_waitcnt', 0)
    p.vop1('v_cvt_f32_u32', 4, 'v0')
    p.vop2('v_mul_f32', 4, ('lit', 0x3d042108), 'v4')
    p.vop1('v_mov_b32', 5, ('lit', FLOAT[0.6]))
    p.vop1('v_mov_b32', 6, ('lit', FLOAT[0.2]))
    p.vop1('v_mov_b32', 7, ('lit', FLOAT[1.0]))
    p.vop1('v_mov_b32', 1, 0)
    p.mimg('image_store', 4, 0, 8)
    p.sopp('s_endpgm')
    r = T.run(p.bytes(), {0: (256, None)}, [], extra=["image 1 32 1 format 56", "timg 0 0 1"])
    got = _image(r, 1, 'B', 128)
    lane_scale = struct.unpack('<f', struct.pack('<I', 0x3d042108))[0]
    for lane in range(32):
        want = (round(lane * lane_scale * 255), 153, 51, 255)
        texel = got[lane * 4:lane * 4 + 4]
        assert all(abs(a - b) <= 1 for a, b in zip(texel, want)), ("lane", lane, texel, want)


def test_store_r32_uint():
    p = rdna_asm.Program()
    p.smem('s_load_dwordx8', 8, 4, 0)
    p.sopp('s_waitcnt', 0)
    p.vop2('v_mul_u32_u24', 4, 3, 'v0')
    p.vop2('v_add_nc_u32', 4, 7, 'v4')
    p.vop1('v_mov_b32', 1, 0)
    p.mimg('image_store', 4, 0, 8, dmask=1)
    p.sopp('s_endpgm')
    r = T.run(p.bytes(), {0: (256, None)}, [], extra=["image 1 32 1 format 20 clear 99 99 99 99", "timg 0 0 1"])
    assert _image(r, 1, 'I', 32) == tuple(lane * 3 + 7 for lane in range(32))


def _load_to_buffer(dim, coords, image_line, image_init):
    p = rdna_asm.Program()
    p.smem('s_load_dwordx8', 8, 4, 0)
    _out_vsharp(p, 16, 32)
    p.sopp('s_waitcnt', 0)
    coords(p)
    p.mimg('image_load', 4, 1, 8, dmask=1, dim=dim)
    p.sopp('s_waitcnt', 0)
    p.vop2('v_lshlrev_b32', 5, 2, 'v0')
    p.mubuf('buffer_store_dword', 4, 5, 16, 128, 0)
    p.sopp('s_endpgm')
    init = os.path.join(T.tempfile.mkdtemp(prefix="img_"), "init.bin")
    open(init, 'wb').write(image_init)
    r = T.run(p.bytes(), {0: (256, None), 2: (128, None)}, T.vsharp(0, 32, 2, 128), extra=[f"{image_line} init {init}", "timg 0 0 1"])
    return struct.unpack('<32I', r[2][:128])


def test_load_3d_r32_uint():
    def coords(p):
        p.vop2('v_and_b32', 1, 3, 'v0')
        p.vop2('v_lshrrev_b32', 2, 2, 'v0')
        p.vop2('v_and_b32', 2, 3, 'v2')
        p.vop2('v_lshrrev_b32', 3, 4, 'v0')
    texels = struct.pack('<32I', *[x + 10 * y + 100 * z for z in range(2) for y in range(4) for x in range(4)])
    got = _load_to_buffer(DIM_3D, coords, "image 1 4 4 format 20 dim 3d depth 2", texels)
    want = tuple((lane & 3) + 10 * ((lane >> 2) & 3) + 100 * (lane >> 4) for lane in range(32))
    assert got == want, (got[:8], want[:8])


def test_load_1d_r32_uint():
    def coords(p):
        p.vop2('v_sub_nc_u32', 1, 31, 'v0')
    texels = struct.pack('<32I', *[5000 + i for i in range(32)])
    got = _load_to_buffer(DIM_1D, coords, "image 1 32 1 format 20 dim 1d", texels)
    assert got == tuple(5000 + 31 - lane for lane in range(32)), got[:8]


def _sample(u, clamp, filt):
    p = rdna_asm.Program()
    p.smem('s_load_dwordx8', 8, 4, 0)
    p.smem('s_load_dwordx4', 16, 4, 32)
    _out_vsharp(p, 20, 48)
    p.sopp('s_waitcnt', 0)
    p.vop1('v_mov_b32', 1, ('lit', FLOAT[u]))
    p.vop1('v_mov_b32', 2, ('lit', FLOAT[0.5]))
    p.mimg('image_sample_lz', 4, 1, 8, dmask=1, ssamp=16)
    p.sopp('s_waitcnt', 0)
    p.vop2('v_lshlrev_b32', 5, 2, 'v0')
    p.mubuf('buffer_store_dword', 4, 5, 20, 128, 0)
    p.sopp('s_endpgm')
    init = os.path.join(T.tempfile.mkdtemp(prefix="img_"), "init.bin")
    open(init, 'wb').write(struct.pack('<2f', 0.0, 1.0))
    ssharp = [clamp | (clamp << 3) | (clamp << 6), 0xfff << 12, (filt << 20) | (filt << 22), 0]
    pokes = [f"poke 0 {32 + 4 * k} {w}" for k, w in enumerate(ssharp)] + T.vsharp(0, 48, 2, 128)
    r = T.run(p.bytes(), {0: (256, None), 2: (128, None)}, pokes, extra=[f"image 1 2 1 format 22 init {init}", "timg 0 0 1"])
    return struct.unpack('<32f', r[2][:128])


def test_resinfo_reports_the_image_extent():
    p = rdna_asm.Program()
    p.smem('s_load_dwordx8', 8, 4, 0)
    _out_vsharp(p, 16, 32)
    p.sopp('s_waitcnt', 0)
    p.vop1('v_mov_b32', 1, 0)
    p.mimg('image_get_resinfo', 4, 1, 8, dmask=3)
    p.sopp('s_waitcnt', 0)
    p.vop2('v_lshlrev_b32', 6, 3, 'v0')
    p.mubuf('buffer_store_dwordx2', 4, 6, 16, 128, 0)
    p.sopp('s_endpgm')
    r = T.run(p.bytes(), {0: (256, None), 2: (256, None)}, T.vsharp(0, 32, 2, 256), extra=["image 1 24 5 format 20", "timg 0 0 1"])
    got = struct.unpack('<64I', r[2][:256])
    assert got == (24, 5) * 32, got[:4]


def test_sampler_from_guest_ssharp():
    WRAP, CLAMP_LAST = 0, 2
    POINT, BILINEAR = 0, 1
    cases = [((1.25, WRAP, POINT), 0.0), ((1.25, CLAMP_LAST, POINT), 1.0), ((0.5, CLAMP_LAST, POINT), 1.0), ((0.5, CLAMP_LAST, BILINEAR), 0.5)]
    for (u, clamp, filt), want in cases:
        got = _sample(u, clamp, filt)
        assert all(abs(g - want) < 1e-3 for g in got), ("u", u, "clamp", clamp, "filter", filt, "got", got[:4], "want", want)


TESTS = [test_resinfo_reports_the_image_extent, test_store_rgba8_unorm, test_store_r32_uint, test_load_3d_r32_uint, test_load_1d_r32_uint, test_sampler_from_guest_ssharp]

if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t(); print("PASS", t.__name__)
        except Exception as e:
            bad += 1; print("FAIL", t.__name__, str(e)[:1500])
    sys.exit(1 if bad else 0)
