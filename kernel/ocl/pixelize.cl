#include "kernel/ocl/common.cl"

#ifdef PARAM
#define PIX_BLOC PARAM
#else
#define PIX_BLOC 16
#endif

// In this over-simplified kernel, all the pixels of a bloc adopt the color
// on the top-left pixel (i.e. we do not compute the average color).
__kernel void pixelize_ocl (__global unsigned *in)
{
  __local unsigned couleur [GPU_TILE_H / PIX_BLOC][GPU_TILE_W / PIX_BLOC];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  if (xloc % PIX_BLOC == 0 && yloc % PIX_BLOC == 0)
    couleur [yloc / PIX_BLOC][xloc / PIX_BLOC] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  in [y * DIM + x] = couleur [yloc / PIX_BLOC][xloc / PIX_BLOC];
}

// Optimized kernel for pixelization.
__kernel void pixelize_ocl_opt (__global unsigned *in)
{
  __local int4 tile [GPU_TILE_H * GPU_TILE_W];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int loc = get_local_id (1) * GPU_TILE_W + get_local_id (0);

  tile [loc] = color_to_int4 (in [y * DIM + x]);
  
  for (int d = (GPU_TILE_W * GPU_TILE_H) >> 1; d > 0; d >>= 1) {
    barrier (CLK_LOCAL_MEM_FENCE);
    
    if (loc < d)
      tile [loc] += tile [loc + d];
  }
  
  barrier (CLK_LOCAL_MEM_FENCE);
  
  in [y * DIM + x] = int4_to_color (tile [0] / (int4) (GPU_TILE_W * GPU_TILE_H));
}
