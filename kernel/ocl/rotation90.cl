#include "kernel/ocl/common.cl"

__kernel void rotation90_ocl (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [(DIM - x - 1) * DIM + y] = in [y * DIM + x];
}

__kernel void rotation90_ocl_opt (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile[GPU_TILE_H][GPU_TILE_W];

  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile [GPU_TILE_W - xloc - 1][yloc] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  out [(DIM - (GPU_TILE_W + x - xloc) + yloc) * DIM + (y - yloc) + xloc] = tile [yloc][xloc];
}

__kernel void rotation90_ocl_magic (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile[GPU_TILE_H][GPU_TILE_W+1];

  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile [GPU_TILE_W - xloc - 1][yloc] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  out [(DIM - (GPU_TILE_W + x - xloc) + yloc) * DIM + (y - yloc) + xloc] = tile [yloc][xloc];
}