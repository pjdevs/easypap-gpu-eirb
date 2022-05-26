#include "kernel/ocl/common.cl"

__kernel void life_ocl (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  if (y > 0 && y < DIM - 1 && x > 0 && x < DIM - 1) {
    unsigned n  = 0;
    unsigned me = in[y * DIM + x];

    for (int yloc = y - 1; yloc < y + 2; yloc++)
      for (int xloc = x - 1; xloc < x + 2; xloc++)
        n += in[yloc * DIM + xloc];

    n = (n == 3 + me) | (n == 3);

    out[y * DIM + x] = n;
  }
}

__kernel void life_ocl_lazy (__global unsigned *in, __global unsigned *out, __global unsigned *last_changed, __global unsigned *next_changed)
{
  int x = get_global_id (0);
  int y = get_global_id (1);
  int tile_x = get_group_id (0);
  int tile_y = get_group_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  if (y <= 0 && y >= DIM - 1 && x <= 0 && x >= DIM - 1)
    return;

  local unsigned changed;

  if (xloc == 0 && yloc == 0) {
    changed = 0;

    for (yloc = y - 1; yloc < y + 2; ++yloc)
      for (xloc = x - 1; xloc < x + 2; ++xloc)
        if (yloc >= 0 && yloc < (DIM / GPU_TILE_H) && xloc >= 0 && xloc < (DIM / GPU_TILE_W))
          changed |= last_changed[yloc * (DIM / GPU_TILE_H) + xloc];

    changed |= last_changed[tile_y * (DIM / GPU_TILE_H) + tile_x];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (!changed) {
    return;
  }

  unsigned n  = 0;
  unsigned me = in[y * DIM + x];

  for (yloc = y - 1; yloc < y + 2; yloc++)
    for (xloc = x - 1; xloc < x + 2; xloc++)
      n += in[yloc * DIM + xloc];

  n = (n == 3 + me) | (n == 3);

  out[y * DIM + x] = n;

  volatile __global unsigned* changed_ptr = next_changed + tile_y * (DIM / GPU_TILE_H) + tile_x; 
  atomic_or(changed_ptr, n != me);
}

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a life-specific version (generic version is defined in common.cl)
__kernel void life_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);

  write_imagef (tex, (int2)(x, y), color_scatter (cur [y * DIM + x] * 0xFFFF00FF));
}