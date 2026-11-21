# tools/xbox

Experimental tooling for running Xbox Series shader code through the shader recompiler. See `docs/xbox/XBOX_SHADERS.md`.

1. `build_msvc.py` builds the recompiler and the tools below with MSVC (run it from PowerShell/cmd). On Linux,
   `cmake -S tools/xbox -B build-xbox -DCMAKE_BUILD_TYPE=Release && cmake --build build-xbox` builds `driver`, `stages` and `exec`
   (needs the `SPIRV-Headers` and `Vulkan-Headers` submodules; `exec` opens the Vulkan loader at run time, as the AGC driver does).
   When a device address does not fit in the 48 bits of a guest pointer (RADV hands out such addresses), `exec` gives the regions remapped
   guest addresses as `lowguest` does. Without `rdna2.py`, `rdna_asm.py` takes its opcode
   tables from LLVM (`rdna2_llvm.py`, needs `llvm-mc` with the AMDGPU target), so `exec_tests.py` and `exec_bvh_tests.py` run with no
   game files: `XBOX_EXEC=build-xbox/exec python3 tools/xbox/exec_tests.py`. `VK_DRIVER_FILES=<icd.json>` picks the GPU.
2. `extract_shaders.py` writes prologue-stripped, trimmed code blobs `<idx>.bin` (it depends on the RPF9
   extraction pipeline and `rdna2.py` decoder kept next to the dumped shaders; adjust its paths).
3. `driver.exe <code.bin> <wave 32|64> <out.spv> [user_data_dwords]` recompiles one compute shader with stubbed
   bindings (user-data pointer pairs backed by fake buffer-descriptor memory).
4. `run_batch.py` runs the driver over a directory of blobs and validates with `spirv-val`.
5. `stages.exe <code.bin>` times decode / CFG build / structurize separately (for the slow ray-tracing shaders).
6. `exec.exe <scenario.txt>` is the execution harness: it recompiles a compute shader against REAL Vulkan device memory, dispatches it on the
   host GPU and writes the memory regions (and images) back. Build it on Windows with `build_msvc.py --with-exec`.
   `rdna_asm.py` assembles small RDNA2 test programs (opcode numbers come from the decoder tables; each program is decoded back to check it),
   `exec_tests.py`, `exec_bvh_tests.py` and `image_tests.py` hold the tests. Set `XBOX_EXEC` to the `exec.exe` path. Supported: guest
   buffers, shader data, flattened SRT, identity BDA page table and fault reporting, single-level linear 1D, 2D, 2D array and 3D images in
   any format the AGC driver maps and the device can sample (`image <id> <w> <h> format <n> dim 1d|2d|3d depth <d>`), T#s encoded with the
   AGC driver's `EncodeTextureResource`, samplers built from the guest S# with its `DecodeSamplerResource`, pixel shaders (`stage ps`).
   Scenario lines added for this: `image <id> <w> <h> [format <IMG_FORMAT>] [dim 1d|2d|3d] [depth <d>] [layers <n>] [clear r g b a]
   [init <file>]` (default format 77, 32_32_32_32_FLOAT; init files hold raw texels in the image format, row-major, slice after slice;
   clear values are integers for UINT/SINT formats; read back as `out/image<id>.bin` in the image format) and `timg <region> <offset>
   <image>`, which writes the image's linear T#. Not yet: block-compressed and cube images, mip levels, image tables, texel buffers,
   other graphics stages. On Linux `exec` links six
   AGC driver and libc sources for this (see `CMakeLists.txt`); an MSVC build of `exec` needs the same files.

## Pixel smoke test

With the Vulkan SDK's `glslangValidator` on PATH and the execution harness built,
run this from the repository root:

```powershell
python tools/xbox/pixel_smoke.py --exec "build-xbox/exec.exe"
```

The vertex shader draws a full-screen triangle. The standalone RDNA2 pixel shader
writes the pixel-center X and Y coordinates, zero blue, and unit alpha to a
13 by 7 RGBA32F attachment. The test checks every channel at every coordinate
and reports the first mismatch. It uses no external shader corpus. The scenario's `export 0 9 0xe4` selects
the 32-bit ABGR export mode with the identity component mapping.

## Vertex color interpolation test

With a built execution harness and the Vulkan SDK's `glslangValidator` on PATH:

```powershell
python tools/xbox/interpolation_test.py --exec "build-xbox/exec.exe"
```

A GLSL vertex shader assigns red, green and blue to a triangle. Its vertices
have clip-space W values of 1, 2 and 4. The RDNA2 fragment program interpolates
each channel using VINTRP P1/P2 and exports RGBA32F. At every pixel center, the
CPU derives screen-space weights from the projected triangle, divides each by
the corresponding W, then normalizes their sum to one. Covered pixels must match
within 0.00001 per channel. Outside pixels must exactly retain the clear value.
The triangle edges avoid pixel centers, so the reference needs no edge tie-break.

## Windows compute smoke test

From the repository root in PowerShell, with Python 3, the x64 MSVC C++ toolchain,
the Vulkan SDK, and a Vulkan-capable GPU driver installed, run:

```powershell
git submodule update --init 3rdparty/SPIRV-Headers
python tools/xbox/build_msvc.py --with-exec
python tools/xbox/exec_smoke.py
```

The build reads `VULKAN_SDK`. Use `--vulkan-sdk "path to SDK"` to select another
installation. For a custom build directory, pass `--exec "path to exec.exe"` to
the smoke test. It generates a standalone RDNA2 compute program, dispatches one
32-lane workgroup, and checks all 1024 output bytes. A successful run prints
`PASS`. This test requires neither a shader corpus nor an external decoder.

### Test texture sampling

With the Vulkan execution harness built and the SDK's `glslangValidator` on PATH,
run `python tools/xbox/texture_test.py --exec "build-xbox/exec.exe"`.

A GLSL vertex shader supplies UVs and unequal clip-space W. An RDNA2 fragment
program interpolates U and V, samples a generated 4 by 3 RGBA32F texture at mip
level zero, and exports the result. The CPU calculates perspective-correct UVs
at each pixel center and independently selects the nearest texel with clamp-to-edge
addressing. All 12 texels must appear, all 192 covered pixels must match exactly,
and all 832 outside pixels must retain the clear value.

Use `python tools/xbox/texture_test.py --bilinear` to select linear minification
and magnification in the guest sampler. This uses the same triangle, mip level,
and texture as the nearest test. The CPU shifts UVs to texel centers, clamps the
four neighbors separately, and weights their RGBA values. The absolute channel
tolerance is 0.002 to allow for finite filter-weight precision. The background
must still match the clear value exactly.

To build the guest-sampler runner on Windows, use an x64 Visual Studio developer
shell with LLVM's clang-cl, CMake, Ninja, and the Vulkan SDK tools on PATH:

```powershell
git submodule update --init 3rdparty/SPIRV-Headers 3rdparty/Vulkan-Headers
cmake -S tools/xbox -B build-sampler -G Ninja -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release
cmake --build build-sampler --target exec
python tools/xbox/texture_test.py --bilinear --exec build-sampler/exec.exe
```
