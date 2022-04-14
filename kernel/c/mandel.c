
#include "easypap.h"

#include <omp.h>


static unsigned compute_one_pixel (int i, int j);
static void zoom (void);


int mandel_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = compute_one_pixel (i, j);

  return 0;
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run --kernel mandel
//
unsigned mandel_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile (0, 0, DIM, DIM, 0);

    zoom ();
  }

  return 0;
}


///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline:
// ./run -k mandel -v tiled -ts 64
//
unsigned mandel_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, 0);

    zoom ();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline:
// ./run -k mandel -v tiled -ts 64
//
unsigned mandel_compute_omp_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    #pragma omp parallel for schedule(runtime) collapse(2)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());

    zoom();
  }

  return 0;
}

/////////////// Mandelbrot basic computation

#define MAX_ITERATIONS 4096
#define ZOOM_SPEED -0.01

static float leftX   = -0.2395;
static float rightX  = -0.2275;
static float topY    = .660;
static float bottomY = .648;

static float xstep;
static float ystep;

void mandel_init ()
{
  // check tile size's conformity with respect to CPU vector width
  // easypap_check_vectorization (VEC_TYPE_FLOAT, DIR_HORIZONTAL);

  xstep = (rightX - leftX) / DIM;
  ystep = (topY - bottomY) / DIM;
}

static unsigned iteration_to_color (unsigned iter)
{
  unsigned r = 0, g = 0, b = 0;

  if (iter < MAX_ITERATIONS) {
    if (iter < 64) {
      r = iter * 2; /* 0x0000 to 0x007E */
    } else if (iter < 128) {
      r = (((iter - 64) * 128) / 126) + 128; /* 0x0080 to 0x00C0 */
    } else if (iter < 256) {
      r = (((iter - 128) * 62) / 127) + 193; /* 0x00C1 to 0x00FF */
    } else if (iter < 512) {
      r = 255;
      g = (((iter - 256) * 62) / 255) + 1; /* 0x01FF to 0x3FFF */
    } else if (iter < 1024) {
      r = 255;
      g = (((iter - 512) * 63) / 511) + 64; /* 0x40FF to 0x7FFF */
    } else if (iter < 2048) {
      r = 255;
      g = (((iter - 1024) * 63) / 1023) + 128; /* 0x80FF to 0xBFFF */
    } else {
      r = 255;
      g = (((iter - 2048) * 63) / 2047) + 192; /* 0xC0FF to 0xFFFF */
    }
  }
  return rgba (r, g, b, 255);
}

static void zoom (void)
{
  float xrange = (rightX - leftX);
  float yrange = (topY - bottomY);

  leftX += ZOOM_SPEED * xrange;
  rightX -= ZOOM_SPEED * xrange;
  topY -= ZOOM_SPEED * yrange;
  bottomY += ZOOM_SPEED * yrange;

  xstep = (rightX - leftX) / DIM;
  ystep = (topY - bottomY) / DIM;
}

static unsigned compute_one_pixel (int i, int j)
{
  float cr = leftX + xstep * j;
  float ci = topY - ystep * i;
  float zr = 0.0, zi = 0.0;

  int iter;

  // Pour chaque pixel, on calcule les termes d'une suite, et on
  // s'arrête lorsque |Z| > 2 ou lorsqu'on atteint MAX_ITERATIONS
  for (iter = 0; iter < MAX_ITERATIONS; iter++) {
    float x2 = zr * zr;
    float y2 = zi * zi;

    /* Stop iterations when |Z| > 2 */
    if (x2 + y2 > 4.0)
      break;

    float twoxy = (float)2.0 * zr * zi;
    /* Z = Z^2 + C */
    zr = x2 - y2 + cr;
    zi = twoxy + ci;
  }

  return iteration_to_color (iter);
}


///////////////////////////////////////////////////////////////////////////
// Copy and paste at the end of mandel.c

// Intrinsics functions
#ifdef ENABLE_VECTO
#include <immintrin.h>

#if __AVX2__ == 1

void mandel_tile_check_avx (void)
{
  // Tile width must be larger than AVX vector size
  easypap_vec_check (AVX_VEC_SIZE_FLOAT, DIR_HORIZONTAL);
}


int mandel_do_tile_avx (int x, int y, int width, int height)
{
  __m256 zr, zi, cr, ci, norm; //, iter;
  __m256 deux     = _mm256_set1_ps (2.0);
  __m256 max_norm = _mm256_set1_ps (4.0);
  __m256i un      = _mm256_set1_epi32 (1);

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j += AVX_VEC_SIZE_FLOAT) {

      __m256i iter = _mm256_setzero_si256 ();

      zr = zi = norm = _mm256_set1_ps (0);

      cr = _mm256_add_ps (_mm256_set1_ps (j),
                          _mm256_set_ps (7, 6, 5, 4, 3, 2, 1, 0));

      cr = _mm256_fmadd_ps (cr, _mm256_set1_ps (xstep), _mm256_set1_ps (leftX));

      ci = _mm256_set1_ps (topY - ystep * i);

      for (int it = 0; it < MAX_ITERATIONS; it++) {
        // rc = zr^2
        __m256 rc = _mm256_mul_ps (zr, zr);

        // |Z|^2 = (partie réelle)^2 + (partie imaginaire)^2 = zr^2 + zi^2
        norm = _mm256_fmadd_ps (zi, zi, rc);

        // On compare les normes au carré de chacun des 8 nombres Z avec 4
        // (normalement on doit tester |Z| <= 2 mais c'est trop cher de calculer
        //  une racine carrée)
        // Le résultat est un vecteur d'entiers (mask) qui contient FF quand
        // c'est vrai et 0 sinon
        __m256i mask = (__m256i)_mm256_cmp_ps (norm, max_norm, _CMP_LE_OS);

        // Il faut sortir de la boucle lorsque le masque ne contient que
        // des zéros (i.e. tous les Z ont une norme > 2, donc la suite a
        // divergé pour tout le monde)

        // FIXME 1
        if (_mm256_testz_si256(mask, mask))
          break;

        // On met à jour le nombre d'itérations effectuées pour chaque pixel.

        // FIXME 2
        iter = _mm256_add_epi32 (iter, _mm256_and_si256(mask, un)); // iter = _mm256_add_epi32 (iter, un);

        // On calcule Z = Z^2 + C et c'est reparti !
        __m256 x = _mm256_add_ps (rc, _mm256_fnmadd_ps (zi, zi, cr));
        __m256 y = _mm256_fmadd_ps (deux, _mm256_mul_ps (zr, zi), ci);
        zr       = x;
        zi       = y;
      }

      cur_img (i, j + 0) = iteration_to_color (_mm256_extract_epi32 (iter, 0));
      cur_img (i, j + 1) = iteration_to_color (_mm256_extract_epi32 (iter, 1));
      cur_img (i, j + 2) = iteration_to_color (_mm256_extract_epi32 (iter, 2));
      cur_img (i, j + 3) = iteration_to_color (_mm256_extract_epi32 (iter, 3));
      cur_img (i, j + 4) = iteration_to_color (_mm256_extract_epi32 (iter, 4));
      cur_img (i, j + 5) = iteration_to_color (_mm256_extract_epi32 (iter, 5));
      cur_img (i, j + 6) = iteration_to_color (_mm256_extract_epi32 (iter, 6));
      cur_img (i, j + 7) = iteration_to_color (_mm256_extract_epi32 (iter, 7));
    }

  return 0;
}

#endif // AVX

#endif
///////////////////////////////////////////////////////////////////////////

