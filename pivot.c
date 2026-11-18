#include "selection.h"
#include <math.h>
// PIVOT SELECTION

/* xorshift64*: small per-rank PRNG with a full 64-bit range. */
uint64_t rng_next(uint64_t *state)
{

    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static int cmp32(const void *a, const void *b)
{
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}
// WINDOWS (type: see selection.h)

void windows_init(Windows *w, int64_t m, int T)
{

    free(w->start);
    int64_t C = (int64_t)CHUNKS_PER_THREAD * T;
    if (m / CHUNK_MIN < C)
        C = m / CHUNK_MIN;
    if (C < 1)
        C = 1;
    w->C = (int)C;
    w->start = xmalloc((6 * C + 2) * sizeof(int64_t));
    w->lo = w->start + (C + 1);
    w->hi = w->lo + C;
    w->lt = w->hi + C;
    w->gt = w->lt + C;
    w->boff = w->gt + C;
    for (int64_t c = 0; c <= C; c++)
        w->start[c] = m * c / C;
    for (int64_t c = 0; c < C; c++)
    {

        w->lo[c] = w->start[c];
        w->hi[c] = w->start[c + 1];
    }
    w->active = m;
}

/* Move all windows to the front of data[], in order: data[o, active) holds the
 * active elements contiguously. Safe in place (the destination never passes the
 * source) and needs no extra memory */
void pack_windows(int32_t *data, const Windows *w)
{

    int64_t o = 0;
    for (int c = 0; c < w->C; c++)
    {

        int64_t len = w->hi[c] - w->lo[c];
        if (len > 0 && w->lo[c] != o)
            memmove(data + o, data + w->lo[c], (size_t)len * sizeof(int32_t));
        o += len;
    }
}

/* Block size of the paper: about log2(N), made odd and at least 5. */
int block_size(int64_t N)
{

    int b = 0;
    for (int64_t t = N; t > 1; t >>= 1)
        b++;
    b |= 1;
    return b < 5 ? 5 : b;
}

/* Paper: local median of block medians. The blocks are processed by the OpenMP
 * threads (the paper's "for i = 1 to m/log m pardo"); the elements of a block
 * get reordered, which is harmless. Requires active > 0. */
int32_t pivot_mom(int32_t *data, Windows *w, int B, int32_t *scratch)
{

    const int C = w->C;
    w->boff[0] = 0;
    for (int c = 0; c < C; c++)
        w->boff[c + 1] = w->boff[c] + (w->hi[c] - w->lo[c] + B - 1) / B;

#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < C; c++)
    {

        int64_t nb = w->boff[c + 1] - w->boff[c];
        for (int64_t i = 0; i < nb; i++)
        {

            int64_t base = w->lo[c] + i * B;
            int64_t len = w->hi[c] - base;
            if (len > B)
                len = B;
            scratch[w->boff[c] + i] = quickselect(data + base, len, len / 2);
        }
    }
    int64_t nb = w->boff[C];
    return quickselect(scratch, nb, nb / 2);
}

/* Median of samples random elements of the window (a single random element is
 * too noisy when p is small) */
int32_t pivot_random(const int32_t *data, const Windows *w,
                     uint64_t *rng)
{

    int32_t s[SAMPLES];
    for (int j = 0; j < SAMPLES; j++)
    {

        int64_t r = (int64_t)(rng_next(rng) % (uint64_t)w->active);
        for (int c = 0; c < w->C; c++)
        {
            int64_t len = w->hi[c] - w->lo[c];
            if (r < len)
            {

                s[j] = data[w->lo[c] + r];
                break;
            }
            r -= len;
        }
    }
    insertion_sort(s, SAMPLES);
    return s[SAMPLES / 2];
}

/* Weighted median of the p candidates (value, weight). Ranks with weight 0
 * are ignored. The result has weight >= N/2 on each side. Weighting by window
 * size matters: a rank with 5 elements left must not outvote one with 5M. */
int32_t weighted_median(const int64_t *vw, int *order, int p,
                        int64_t N)
{

    int q = 0;
    for (int r = 0; r < p; r++)
        if (vw[2 * r + 1] > 0)
            order[q++] = r;
    for (int i = 1; i < q; i++)
    { /* insertion sort by candidate value */
        int v = order[i], j = i - 1;
        while (j >= 0 && vw[2 * order[j]] > vw[2 * v])
        {

            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = v;
    }
    int64_t cum = 0;
    for (int i = 0; i < q; i++)
    {

        cum += vw[2 * order[i] + 1];
        if (2 * cum >= N)
            return (int32_t)vw[2 * order[i]];
    }
    return (int32_t)vw[2 * order[q - 1]]; /* unreachable when N > 0 */
}
/*  
    Floyd-Rivest bounds (Floyd & Rivest, CACM 1975). Picks u<=v so that the k-smallest element is very likely in [u,v]
    and few elements are in between: 
    
    1. every rank draws a share of a random sample, proportional to the data it still holds, so the sample is uniform over the surviving elemnts;
    2. the sample is allgathered and sorted (same result on every rank);
    3. u and v are the sample values d*sqrt(s) positions below and above the sample rank that corresponds to k, with d = sqrt(ln N).

    A bound that falls off the sample becomes INT32_MIN / INT32_MAX.
    COLLECTIVE: all ranks must call it with the same N,k, and counts.
*/
void fr_bounds(const int32_t *data, const Windows *w, const int64_t *counts, int64_t N, int64_t k, int p, int me, uint64_t *rng, int32_t *u, int32_t *v)
{
    /* sample size from the paper: s ~ N^(2/3) * ln(N)^(1/3) */
    int64_t s = (int64_t)(pow(N, 2.0 / 3.0) * cbrt(log(N)));
    if (s > N)
        s = N;
    /* each rank's share is computed from counts[]: no extra communication */
    int *rc = xmalloc(2 * p * sizeof(int)), *rd = rc + p;
    int64_t tot = 0;
    for (int r = 0; r < p; r++)
    {
        rc[r] = (int)(s * counts[r] / N);
        rd[r] = (int)tot;
        tot += rc[r];
    }
    s = tot;
    /* nothing to sample: return the widest bracket (the loop then falls back) */
    if (s < 1)
    {
        free(rc);
        *u = INT32_MIN;
        *v = INT32_MAX;
        return;
    }
    /* draw with replacement: pick a random live element, walking the chunks */
    int32_t *all = xmalloc(s * sizeof(int32_t));
    for (int j = 0; j < rc[me]; j++)
    {
        int64_t r = (int64_t)(rng_next(rng) % (uint64_t)w->active);
        for (int c = 0; c < w->C; c++)
        {
            int64_t len = w->hi[c] - w->lo[c];
            if (r < len)
            {
                all[rd[me] + j] = data[w->lo[c] + r];
                break;
            }
            r -= len;
        }
    }
    /* each rank filled its own slice; Allgatherv in place shares the rest */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all, rc, rd, MPI_INT32_T, MPI_COMM_WORLD);
    qsort(all, s, sizeof(int32_t), cmp32);

    /* expected sample rank of k, and a margin of d standard deviations */
    double d = sqrt(log((double)N));
    double center = (double)k * (double)(s + 1) / (double)(N + 1);
    double off = d * sqrt((double)s);
    int64_t ru = (int64_t)floor(center - off);
    int64_t rv = (int64_t)ceil(center + off);
    *u = (ru < 0) ? INT32_MIN : all[ru];
    *v = (rv >= s) ? INT32_MAX : all[rv];

    free(all);
    free(rc);
}