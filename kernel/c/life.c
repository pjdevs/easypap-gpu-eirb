
#include "easypap.h"
#include "rle_lexer.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <assert.h>

static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

typedef unsigned cell_t;

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;
static unsigned *restrict _last_changed = NULL, *restrict _next_changed = NULL;

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))
#define last_changed_table(y, x) (_last_changed[(y) * (DIM / TILE_H) + (x)])
#define next_changed_table(y, x) (_next_changed[(y) * (DIM / TILE_H) + (x)])

void life_init (void)
{
  // life_init may be (indirectly) called several times so we check if data were
  // already allocated
  if (_table == NULL) {
    const unsigned size = DIM * DIM * sizeof (cell_t);
    const unsigned changed_size = (DIM / TILE_H) * (DIM / TILE_W) * sizeof(unsigned);

    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes (classic) + %d bytes (lazy)\n", size, changed_size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);


    _last_changed = mmap (NULL, changed_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    _next_changed = mmap (NULL, changed_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    for (unsigned i = 0; i < (changed_size / sizeof(unsigned)); ++i) {
      _last_changed[i] = 1;
      _next_changed[i] = 0;
    }
  }
}

void life_finalize (void)
{
  const unsigned size = DIM * DIM * sizeof (cell_t);
  const unsigned changed_size = (DIM / TILE_H) * (DIM / TILE_W) * sizeof(unsigned);

  munmap (_table, size);
  munmap (_alternate_table, size);
  munmap (_last_changed, changed_size);
  munmap (_next_changed, changed_size);
}

// This function is called whenever the graphical window needs to be refreshed
void life_refresh_img (void)
{
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      cur_img (i, j) = cur_table (i, j) * color;
}

cl_mem last_changed_buffer = 0, next_changed_buffer = 0;

void life_init_ocl_lazy (void)
{
  life_init();

  const size_t changed_size = (GPU_SIZE_X / GPU_TILE_W) * (GPU_SIZE_Y / GPU_TILE_H) * sizeof (unsigned);

  last_changed_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, changed_size, NULL, NULL);
  if (!last_changed_buffer)
    exit_with_error ("Failed to allocate last_changed_buffer");

  next_changed_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, changed_size, NULL, NULL);  
  if (!next_changed_buffer)
    exit_with_error ("Failed to allocate next_changed_buffer");

  cl_int err;

  unsigned *tmp = malloc(changed_size);

  for (unsigned i = 0; i < changed_size / sizeof(unsigned); ++i)
    tmp[i] = 1;

  err = clEnqueueWriteBuffer (queue, last_changed_buffer, CL_TRUE, 0,
                              changed_size, tmp, 0, NULL, NULL);
  check (err, "Failed to write to last_changed_buffer");

  free(tmp);
}

unsigned life_invoke_ocl_lazy (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
  size_t local[2]  = {GPU_TILE_W, GPU_TILE_H}; // local domain size for our calculation
  cl_int err;

  monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {
    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (cl_mem), &last_changed_buffer);
    err |= clSetKernelArg (compute_kernel, 3, sizeof (cl_mem), &next_changed_buffer);

    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      cl_mem tmp  = cur_buffer;
      cur_buffer = next_buffer;
      next_buffer = tmp;

      tmp = last_changed_buffer;
      last_changed_buffer = next_changed_buffer;
      next_changed_buffer = tmp;
    }
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}

// Only called when --dump or --thumbnails is used
void life_refresh_img_ocl(void)
{
  cl_int err;

  err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                            sizeof(unsigned) * DIM * DIM, _table, 0, NULL,
                            NULL);
  check(err, "Failed to read buffer from GPU");

  life_refresh_img();
}

static inline void swap_tables (void)
{
  cell_t *tmp = _table;

  _table           = _alternate_table;
  _alternate_table = tmp;

  unsigned* t = _last_changed;
  
  _last_changed = _next_changed;
  _next_changed = t;
}

///////////////////////////// Default tiling
int life_do_tile_default (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      if (j > 0 && j < DIM - 1 && i > 0 && i < DIM - 1) {

        unsigned n  = 0;
        unsigned me = cur_table (i, j);

        for (int yloc = i - 1; yloc < i + 2; yloc++)
          for (int xloc = j - 1; xloc < j + 2; xloc++)
            n += cur_table (yloc, xloc);

        n = (n == 3 + me) | (n == 3);
        change |= (n != me);

        next_table (i, j) = n;
      }

  return change;
}

///////////////////////////// Sequential version (seq)
//
unsigned life_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    int change = do_tile (0, 0, DIM, DIM, 0);

    if (!change)
      return it;

    swap_tables ();
  }

  return 0;
}


///////////////////////////// Tiled sequential version (tiled)
//
unsigned life_compute_tiled (unsigned nb_iter)
{
  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |= do_tile (x, y, TILE_W, TILE_H, 0);

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

// omp tiled version
unsigned life_compute_omp_tiled (unsigned nb_iter)
{
  unsigned res = 0, change = 0, temp = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    change = 0;
    temp = 0;

    #pragma omp parallel for schedule(runtime) collapse(2) shared(change) private(temp)
    for (int y = 0; y < DIM; y += TILE_H)
    {
      for (int x = 0; x < DIM; x += TILE_W)
      {
        temp = do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());

        #pragma omp critical
        change |= temp;
      }
    }

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

// One parallel section with barrier and single (a little bit less efficient)
unsigned life_compute_omp_tiled_barrier (unsigned nb_iter)
{
  unsigned res = 0, change = 0, temp = 0;

  #pragma omp parallel shared(change) private(temp)
  for (unsigned it = 1; it <= nb_iter; it++) {
    change = 0;
    temp = 0;

    #pragma omp for schedule(runtime) collapse(2)
    for (int y = 0; y < DIM; y += TILE_H)
    {
      for (int x = 0; x < DIM; x += TILE_W)
      {
        temp = do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());

        #pragma omp atomic
        change |= temp;
      }
    }

    #pragma omp barrier

    #pragma omp single
    swap_tables ();
  }

  return res;
}

// One parallel section with one thread gving task to others and a barrier before swap
unsigned life_compute_omp_tiled_task (unsigned nb_iter)
{
  unsigned res = 0, change = 0, temp = 0;

  #pragma omp parallel shared(change, res) private(temp)
  #pragma omp single
  for (unsigned it = 1; it <= nb_iter; it++) {
    change = 0;
    temp = 0;

    for (int y = 0; y < DIM; y += TILE_H)
    {
      for (int x = 0; x < DIM; x += TILE_W)
      {
        #pragma omp task
        {
          temp = do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
          #pragma omp atomic
          change |= temp;
        }
      }
    }

    #pragma omp taskwait

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      it = nb_iter;
    }
  }

  return res;
}

unsigned tiles_around_changed(int x, int y)
{
  if (last_changed_table(y, x))
    return 1;

  unsigned changed = 0;

  for (int yloc = y - 1; yloc < y + 2; ++yloc)
    for (int xloc = x - 1; xloc < x + 2; ++xloc)
      if (yloc >= 0 && yloc < (DIM / TILE_H) && xloc >= 0 && xloc < (DIM / TILE_W))
        changed |= last_changed_table(yloc, xloc);

  return changed;
}

// omp tiled version
unsigned life_compute_omp_tiled_lazy (unsigned nb_iter)
{
  unsigned res = 0, change = 0, temp = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    change = 0;
    temp = 0;

    #pragma omp parallel for schedule(runtime) collapse(2) shared(change) private(temp)
    for (int y = 0; y < DIM; y += TILE_H)
    {
      for (int x = 0; x < DIM; x += TILE_W)
      {
        temp = tiles_around_changed(x / TILE_W, y / TILE_H);

        if (temp)
          temp = do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());

        next_changed_table(y / TILE_H, x / TILE_W) = temp;

        #pragma omp atomic
        change |= temp;
      }
    }

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

///////////////////////////// Initial configs

void life_draw_guns (void);

static inline void set_cell (int y, int x)
{
  cur_table (y, x) = 1;
  if (opencl_used)
    cur_img (y, x) = 1;
}

static inline int get_cell (int y, int x)
{
  return cur_table (y, x);
}

static void inline life_rle_parse (char *filename, int x, int y,
                                   int orientation)
{
  rle_lexer_parse (filename, x, y, set_cell, orientation);
}

static void inline life_rle_generate (char *filename, int x, int y, int width,
                                      int height)
{
  rle_generate (x, y, width, height, get_cell, filename);
}

void life_draw (char *param)
{
  if (param && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a RLE-encoded file
    life_rle_parse (param, 1, 1, RLE_ORIENTATION_NORMAL);
  } else
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper (param, life_draw_guns);
}

static void otca_autoswitch (char *name, int x, int y)
{
  life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_rle_parse ("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void otca_life (char *name, int x, int y)
{
  life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_rle_parse ("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners (char *filename, int distance)
{
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_NORMAL);
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_HINVERT);
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_VINVERT);
  life_rle_parse (filename, distance, distance,
                  RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_off -ts 64 -r 10 -si
void life_draw_otca_off (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_on -ts 64 -r 10 -si
void life_draw_otca_on (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_draw_meta3x3 (void)
{
  if (DIM < 6208)
    exit_with_error ("DIM should be at least %d", 6208);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      otca_life (j == 1 ? "data/rle/otca-on.rle" : "data/rle/otca-off.rle",
                 1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life -a bugs -ts 64
void life_draw_bugs (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }
}

// Suggested cmdline: ./run -k life -v omp -a ship -s 512 -m -ts 16
void life_draw_ship (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }

  for (int y = 43; y < DIM - 134; y += 148) {
    life_rle_parse ("data/rle/greyship.rle", DIM - 100, y,
                    RLE_ORIENTATION_NORMAL);
  }
}

void life_draw_stable (void)
{
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4) {
      set_cell (i, j);
      set_cell (i, j + 1);
      set_cell (i + 1, j);
      set_cell (i + 1, j + 1);
    }
}

void life_draw_guns (void)
{
  at_the_four_corners ("data/rle/gun.rle", 1);
}

void life_draw_random (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      if (random () & 1)
        set_cell (i, j);
}

// Suggested cmdline: ./run -k life -a clown -s 256 -i 110
void life_draw_clown (void)
{
  life_rle_parse ("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}

void life_draw_diehard (void)
{
  life_rle_parse ("data/rle/diehard.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}
