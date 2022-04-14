#include "kernel/ocl/common.cl"


__kernel void sample_ocl (__global unsigned *img)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  if ((get_group_id (0) + get_group_id (1)) % 2)
    img [y * DIM + x] = 0xFFFF00FF;
  else
    img [y * DIM + x] = 0xF00F00FF;
}
