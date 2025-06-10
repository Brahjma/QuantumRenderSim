```c
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <errno.h>
#include <assert.h>

// --- Configuration ---
#define NUM_QUBITS 16
#define NUM_QPUS 2
#define NUM_RAYS 1024
#define NUM_TRIANGLES 32
#define NUM_LIGHTS 4
#define NUM_THREADS 8
#define PAGE_SIZE 4096
#define EVENT_SIZE 64
#define RING_SIZE 1024
#define BATCH_SIZE 16
#define BATCH_TIMEOUT_US 1000
#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define MAX_CYCLES 5
#define BACKOFF_MAX_US 10000
#define MAX_ATTEMPTS 10
#define MAX_WAIT_COUNT 1000

// --- VMM ---
/** Simulated virtual memory context. */
typedef struct { int64_t pid; } VirtualMemoryContext;

/** Pthread mutex for thread synchronization. */
typedef pthread_mutex_t simple_lock_t;

/** @brief Initializes a mutex lock. */
static inline void simple_lock_init(simple_lock_t *lock) {
    if (pthread_mutex_init(lock, NULL) != 0) {
        syslog(LOG_ERR, "Failed to initialize mutex: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }
}

/** @brief Acquires a mutex lock. */
static inline void simple_lock(simple_lock_t *lock) { pthread_mutex_lock(lock); }

/** @brief Releases a mutex lock. */
static inline void simple_unlock(simple_lock_t *lock) { pthread_mutex_unlock(lock); }

/** @brief Destroys a mutex lock. */
static inline void simple_lock_destroy(simple_lock_t *lock) { pthread_mutex_destroy(lock); }

/** @brief Initializes the virtual memory manager. */
int vmm_init(void) {
    syslog(LOG_INFO, "VMM: Initialized");
    return 0;
}

/** @brief Allocates a process context. */
VirtualMemoryContext* vmm_allocate_process_context(int64_t pid) {
    VirtualMemoryContext *ctx = malloc(sizeof(VirtualMemoryContext));
    if (ctx != NULL) {
        ctx->pid = pid;
        syslog(LOG_INFO, "VMM: Allocated context for PID %ld", pid);
    } else {
        syslog(LOG_ERR, "VMM: Failed to allocate context for PID %ld", pid);
    }
    return ctx;
}

/** @brief Maps a page-aligned memory region. */
unsigned long long vmm_map_page(VirtualMemoryContext *ctx, unsigned int vpn, bool writable, bool user_access) {
    void *ptr = aligned_alloc(PAGE_SIZE, PAGE_SIZE);
    if (ptr != NULL) {
        syslog(LOG_DEBUG, "VMM: Mapped page for VPN %u to %p (PID %ld)", vpn, ptr, ctx ? ctx->pid : -1);
        return (unsigned long long)ptr;
    }
    syslog(LOG_ERR, "VMM: Failed to allocate page for VPN %u (PID %ld)", vpn, ctx ? ctx->pid : -1);
    return 0;
}

/** @brief Dummy page allocation for buddy system. */
int buddy_alloc(int order) { return rand() % 1000; }

/** @brief Dummy page deallocation for buddy system. */
void buddy_free(int frame_idx, int order) {
    syslog(LOG_DEBUG, "Buddy: Freeing frame %d (order %d)", frame_idx, order);
}

// --- Ring Buffer ---
/** @brief Event structure for the ring buffer. */
typedef struct {
    uint64_t event_id;
    int64_t thread_id;
    unsigned int32_t ray_idx;
    uint64_t scene_id;
    float brightness;
    char padding[EVENT_SIZE - sizeof(uint64_t) - sizeof(int64_t) - sizeof(unsigned int32_t) - sizeof(uint64_t) - sizeof(float)];
} Event;

static_assert(sizeof(Event) == EVENT_SIZE, "Event size must be exactly EVENT_SIZE");

/** @brief Lock-free ring buffer for events. */
typedef struct {
    _Alignas(64) atomic_uint64_t head;
    _Alignas(64) atomic_uint64_t tail;
    _Alignas(64_t) Event buffer[RING_SIZE];
} AtomicEventRingBuffer;

/** @brief Initializes the ring buffer. */
void ring_buffer_init(AtomicEventRingBuffer *rb) {
    atomic_store_explicit(&rb->head, 0, memory_order_relaxed);
    atomic_store_explicit(&rb->tail, 0, memory_order_relaxed);
    syslog(LOG_INFO, "RingBuffer: Initialized (%d slots)", RING_SIZE);
}

/** @brief Enqueues an event with exponential backoff. */
int enqueue_event(AtomicEventRingBuffer *rb, Event *event {
    uint64_t backoff_us = 1;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_relaxed);
        uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire);
        if (tail > tail - head >= RING_SIZE) {
            syslog(LOG_DEBUG, "RingBuffer full, attempt %d, backoff %lu us", attempt, backoff_us);
            if (backoff_us > BACKOFF_MAX_US) {
                syslog(LOG_WARNING, "Failed to enqueue event %lu after %d attempts", event->event_id, MAX_ATTEMPTS);
                return -1;
            }
            usleep(backoff_us);
            backoff_us *= 2;
            continue;
        }
        if (atomic_compare_exchange_strong_explicit(&rb->tail, &tail, tail + 1, memory_order_release, memory_order_relaxed)) {
            rb->buffer[tail % RING_SIZE] = *event;
            syslog(LOG_DEBUG, "Enqueued event %lu (ray %u, scene %lu, brightness %.2f)", 
                   event->event_id, event->ray_idx, event->scene_id, event->brightness);
            return 0;
        }
        sched_yield();
    }
    syslog(LOG_WARNING, "Failed to enqueue event %lu after %d attempts", event->event_id, MAX_ATTEMPTS);
    return -1;
}

/** @brief Dequeues an event with exponential backoff. */
int dequeue_event(AtomicEventRingBuffer *rb, Event *event_data) {
    uint64_t backoff_us = 1;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        uint64_t head = atomic_load_explicit(&rb->head, memory_order_relaxed);
        uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);
        if (head == tail) {
            syslog(LOG_DEBUG, "RingBuffer empty, attempt %d, backoff %lu us", attempt, backoff_us);
            if (backoff_us > BACKOFF_MAX_US) {
                syslog(LOG_WARNING, "Failed to dequeue event after %d attempts", MAX_ATTEMPTS);
                return -1;
            }
            usleep(backoff_us);
            backoff_us *= 2;
            continue;
        }
        if (atomic_compare_exchange_strong_explicit(&rb->head, &head, head + 1, memory_order_release, memory_order_relaxed)) {
            *event_data = rb->buffer[head % RING_SIZE];
            syslog(LOG_DEBUG, "Dequeued event %lu (ray %u, scene %lu, brightness %.2f)", 
                   event_data->event_id, event_data->ray_idx, event_data->scene_id, event_data->brightness);
            return 0;
        }
        sched_yield();
    }
    syslog(LOG_WARNING, "Failed to dequeue event after %d attempts", MAX_ATTEMPTS);
    return -1;
}

// --- 3D Structures ---
/** @brief 3D vector structure. */
typedef struct { float x, y, z; } Vector3D;

/** @brief Triangle with vertices and material properties. */
typedef struct { Vector3D v0, v1, v2; float albedo; int texture_id; } Triangle;

/** @brief Light source with position and intensity. */
typedef struct { Vector3D pos; float intensity; } Light;

/** @brief Ray with origin, direction, and intensity. */
typedef struct { Vector3D origin, dir; float intensity; } Ray;

/** @brief Texture with noise parameters from QPU. */
typedef struct { float noise_params[NUM_QUBITS]; int width, height; } Texture;

/** @brief 3D scene with geometry, lights, rays, and textures. */
typedef struct {
    Triangle triangles[NUM_TRIANGLES];
    Light lights[NUM_LIGHTS];
    Ray rays[NUM_RAYS];
    Texture textures[NUM_QPUS];
    uint64_t scene_id;
    int cycle;
} Scene3D;

/** @brief Hierarchical simulation level. */
typedef struct LevelN {
    Scene3D *scene;
    struct QPUOutput *output_combined;
    int level_id;
    struct LevelN *next_level;
} LevelN;

// --- QPU Output ---
/** @brief Output parameters from a QPU. */
typedef struct {
    float surface_albedo[NUM_TRIANGLES];
    float light_sources_intensity[NUM_LIGHTS];
    float noise_seed_from_qpu[NUM_QUBITS];
    uint64_t derived_scene_id;
} QPUOutput;

// --- QPU API ---
/** @brief QPU context for quantum computations. */
typedef struct {
    int qubits;
    float *quantum_states;
    float *entangled_pairs;
    unsigned int rand_seed;
} QPUContext;

/** @brief Quantum circuit for QAOA. */
typedef struct {
    int qubits;
    float *params;
} QPUCircuit;

/** @brief XORShift random number generator. */
static inline unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

/** @brief Initializes a QPU context. */
QPUContext* qpu_init(int qubits, unsigned int initial_seed) {
    QPUContext *ctx = malloc(sizeof(QPUContext));
    if (!ctx) {
        syslog(LOG_ERR, "QPU: Failed to allocate context");
        return NULL;
    }
    ctx->qubits = qubits;
    ctx->quantum_states = malloc(qubits * sizeof(float));
    ctx->entangled_pairs = malloc(qubits * sizeof(float));
    if (!ctx->quantum_states || !ctx->entangled_pairs) {
        syslog(LOG_ERR, "QPU: Failed to allocate states or pairs");
        free(ctx->quantum_states);
        free(ctx->entangled_pairs);
        free(ctx);
        return NULL;
    }
    ctx->rand_seed = initial_seed ^ (unsigned int)time(NULL) ^ (unsigned int)pthread_self();
    for (int i = 0; i < qubits; i++) {
        ctx->quantum_states[i] = 0.707f;
        ctx->entangled_pairs[i] = 0.0f;
    }
    syslog(LOG_INFO, "QPU: Initialized with %d qubits, seed %u", qubits, ctx->rand_seed);
    return ctx;
}

/** @brief Frees a QPU context. */
void qpu_free(QPUContext *ctx) {
    if (ctx) {
        free(ctx->quantum_states);
        free(ctx->entangled_pairs);
        free(ctx);
        syslog(LOG_DEBUG, "QPU: Context freed");
    }
}

/** @brief Creates a QAOA circuit. */
QPUCircuit* qpu_create_circuit(QPUContext *ctx, int qubits) {
    QPUCircuit *circ = malloc(sizeof(QPUCircuit));
    if (!circ) {
        syslog(LOG_ERR, "QPU: Failed to allocate circuit");
        return NULL;
    }
    circ->qubits = qubits;
    circ->params = malloc(qubits * sizeof(float));
    if (!circ->params) {
        syslog(LOG_ERR, "QPU: Failed to allocate circuit params");
        free(circ);
        return NULL;
    }
    for (int i = 0; i < qubits; i++) circ->params[i] = 0.5f;
    syslog(LOG_DEBUG, "QPU: Created QAOA circuit with %d qubits", qubits);
    return circ;
}

/** @brief Applies a quantum gate. */
void qpu_apply_gate(QPUContext *ctx, const char *gate_type, int qubit_idx, float param) {
    if (!ctx || qubit_idx < 0 || qubit_idx >= ctx->qubits) {
        syslog(LOG_WARNING, "QPU: Invalid gate (qubit %d, qubits %d)", qubit_idx, ctx ? ctx->qubits : -1);
        return;
    }
    if (strcmp(gate_type, "H") == 0) {
        ctx->quantum_states[qubit_idx] = 0.707f;
    } else if (strcmp(gate_type, "RX") == 0) {
        ctx->quantum_states[qubit_idx] = cosf(param) * ctx->quantum_states[qubit_idx];
    } else if (strcmp(gate_type, "CNOT") == 0) {
        int target = (qubit_idx + 1) % ctx->qubits;
        ctx->entangled_pairs[qubit_idx] = 0.5f;
    }
    syslog(LOG_DEBUG, "QPU: Applied %s gate to qubit %d (param %.2f)", gate_type, qubit_idx, param);
}

/** @brief Measures a qubit. */
int qpu_measure_qubit(QPUContext *ctx, int qubit_idx) {
    if (!ctx || qubit_idx < 0 || qubit_idx >= ctx->qubits) {
        syslog(LOG_WARNING, "QPU: Invalid measurement (qubit %d, qubits %d)", qubit_idx, ctx ? ctx->qubits : -1);
        return -1;
    }
    float prob = ctx->quantum_states[qubit_idx] * ctx->quantum_states[qubit_idx];
    int result = ((float)xorshift32(&ctx->rand_seed) / (float)UINT_MAX) < prob ? 1 : 0;
    ctx->quantum_states[qubit_idx] = result ? 1.0f : 0.0f;
    syslog(LOG_DEBUG, "QPU: Measured qubit %d -> %d (prob %.2f)", qubit_idx, result, prob);
    return result;
}

/** @brief Executes a QAOA circuit on a scene. */
void qpu_execute(QPUCircuit *circ, QPUContext *ctx, Scene3D *scene, QPUOutput *output, int qpu_id) {
    if (!circ || !ctx || !scene || !output) {
        syslog(LOG_ERR, "QPU %d: Invalid arguments", qpu_id);
        return;
    }
    float costs[NUM_TRIANGLES] = {0};
    float total_light_intensity = 0.0f;
    for (int k = 0; k < NUM_LIGHTS; k++) total_light_intensity += scene->lights[k].intensity;
    int rays_per_qpu = NUM_RAYS / NUM_QPUS;
    int start_ray_idx = qpu_id * rays_per_qpu;
    int end_ray_idx = (qpu_id == NUM_QPUS - 1) ? NUM_RAYS : start_ray_idx + rays_per_qpu;
    for (int i = start_ray_idx; i < end_ray_idx; i++) {
        for (int j = 0; j < NUM_TRIANGLES; j++) {
            float dx = scene->rays[i].origin.x - scene->triangles[j].v0.x;
            float dy = scene->rays[i].origin.y - scene->triangles[j].v0.y;
            float dz = scene->rays[i].origin.z - scene->triangles[j].v0.z;
            float dist_sq = dx * dx + dy * dy + dz * dz + 0.0001f;
            costs[j] += (scene->rays[i].intensity * total_light_intensity * scene->triangles[j].albedo) / dist_sq;
        }
    }
    for (int iter = 0; iter < 5; iter++) {
        for (int i = 0; i < ctx->qubits; i++) {
            qpu_apply_gate(ctx, "H", i, 0.0f);
            float beta = circ->params[i] + iter * 0.1f + scene->cycle * 0.05f;
            qpu_apply_gate(ctx, "RX", i, beta * costs[i % NUM_TRIANGLES]);
            if (i < ctx->qubits - 1) qpu_apply_gate(ctx, "CNOT", i, 0.0f);
        }
    }
    for (int i = 0; i < NUM_TRIANGLES; i++) {
        output->surface_albedo[i] = (float)qpu_measure_qubit(ctx, i % NUM_QUBITS) * 0.8f + 0.2f;
    }
    for (int i = 0; i < NUM_LIGHTS; i++) {
        output->light_sources_intensity[i] = (float)qpu_measure_qubit(ctx, i % NUM_QUBITS) * 5.0f;
    }
    for (int i = 0; i < NUM_QUBITS; i++) {
        output->noise_seed_from_qpu[i] = (float)qpu_measure_qubit(ctx, i);
        scene->textures[qpu_id].noise_params[i] = output->noise_seed_from_qpu[i];
    }
    output->derived_scene_id = scene->scene_id + 1;
    syslog(LOG_INFO, "QPU %d: Generated parameters for scene %lu", qpu_id, output->derived_scene_id);
}

/** @brief Frees a QAOA circuit. */
void qpu_free_circuit(QPUCircuit *circ) {
    if (circ) {
        free(circ->params);
        free(circ);
        syslog(LOG_DEBUG, "QPU: Circuit freed");
    }
}

// --- Perlin Noise ---
/** @brief Precomputed noise table. */
static float noise_table[256];
static bool noise_table_initialized = false;
static pthread_mutex_t noise_table_mutex = PTHREAD_MUTEX_INITIALIZER;

/** @brief Initializes the noise table (thread-safe). */
static void init_noise_table(void) {
    if (!noise_table_initialized) {
        pthread_mutex_lock(&noise_table_mutex);
        if (!noise_table_initialized) {
            unsigned int seed = (unsigned int)time(NULL);
            for (int i = 0; i < 256; i++) {
                noise_table[i] = (float)xorshift32(&seed) / UINT_MAX;
            }
            noise_table_initialized = true;
            syslog(LOG_INFO, "Perlin: Noise table initialized");
        }
        pthread_mutex_unlock(&noise_table_mutex);
    }
}

/** @brief Linear interpolation. */
static inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

/** @brief Generates Perlin noise. */
float perlin_noise(float x, float y, float *noise_params) {
    init_noise_table();
    if (!noise_params) {
        int idx = ((int)(x * 10) % 256 + 256) % 256;
        return noise_table[idx] * sinf(x + y);
    }
    int seed_idx = ((int)x + (int)y) % NUM_QUBITS;
    int ix = (int)(x * 10) % 256; if (ix < 0) ix += 256;
    int iy = (int)(y * 10) % 256; if (iy < 0) iy += 256;
    float fx = (x * 10) - (int)(x * 10);
    float fy = (y * 10) - (int)(y * 10);
    float n00 = noise_table[(ix + iy * 16) % 256] * noise_params[seed_idx];
    float n01 = noise_table[(ix + (iy + 1) * 16) % 256] * noise_params[seed_idx];
    float n10 = noise_table[(ix + 1 + iy * 16) % 256] * noise_params[seed_idx];
    float n11 = noise_table[(ix + 1 + (iy + 1) * 16) % 256] * noise_params[seed_idx];
    float nx0 = lerp(n00, n10, fx);
    float nx1 = lerp(n01, n11, fx);
    return lerp(nx0, nx1, fy) * sinf(x + y);
}

// --- PPM Rendering ---
/** @brief Renders the scene to a PPM file. */
int render_gpu(QPUOutput *output, Scene3D *scene, AtomicEventRingBuffer *rb, int cycle) {
    if (!output || !scene || !rb) {
        syslog(LOG_ERR, "GPU: Invalid arguments");
        return -1;
    }
    char filename[128];
    snprintf(filename, sizeof(filename), "render_cycle_%03d_scene_%lu.ppm", cycle, output->derived_scene_id);
    FILE *f = fopen(filename, "w");
    if (!f) {
        syslog(LOG_ERR, "GPU: Failed to open %s: %s", filename, strerror(errno));
        return -1;
    }
    fprintf(f, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    float total_pixel_value = 0.0f;
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int tri_idx = (x + y) % NUM_TRIANGLES;
            float albedo = output->surface_albedo[tri_idx];
            float tex_noise = perlin_noise(x / 10.0f, y / 10.0f, scene->textures[0].noise_params);
            int r = (int)(albedo * 255 * (1.0f + tex_noise));
            int g = (int)(albedo * 200 * (1.0f + tex_noise));
            int b = (int)(albedo * 150 * (1.0f + tex_noise));
            r = r > 255 ? 255 : r < 0 ? 0 : r;
            g = g > 255 ? 255 : g < 0 ? 0 : g;
            b = b > 255 ? 255 : b < 0 ? 0 : b;
            fprintf(f, "%d %d %d ", r, g, b);
            total_pixel_value += (float)(r + g + b);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    syslog(LOG_INFO, "GPU: Rendered scene %lu (Cycle %d) to %s", output->derived_scene_id, cycle, filename);
    float average_brightness = total_pixel_value / (float)(IMAGE_WIDTH * IMAGE_HEIGHT * 3 * 255);
    if (cycle < MAX_CYCLES - 1) {
        Event event = {
            .event_id = output->derived_scene_id,
            .thread_id = 0,
            .ray_idx = 0,
            .scene_id = output->derived_scene_id,
            .brightness = average_brightness
        };
        if (enqueue_event(rb, &event) != 0) {
            syslog(LOG_WARNING, "GPU: Failed to enqueue brightness event for scene %lu", event.scene_id);
        }
    }
    return 0;
}

// --- Zero Host ---
/** @brief Arguments for QPU threads. */
typedef struct { struct ZeroHost *host; int qpu_id; } QPUArgs;

/** @brief Arguments for worker threads. */
typedef struct { struct ZeroHost *host; int thread_idx; } WorkerArgs;

/** @brief Host managing QPUs and simulation levels. */
typedef struct ZeroHost {
    QPUContext *qpu_ctxs[NUM_QPUS];
    AtomicEventRingBuffer *event_queue;
    LevelN *current_level;
    QPUOutput *output_buffers[NUM_QPUS];
    VirtualMemoryContext *vmm_ctx;
    simple_lock_t lock;
    Event batch[BATCH_SIZE];
    int batch_count;
    uint64_t last_batch_time;
    pthread_t qpu_threads[NUM_QPUS];
    QPUArgs qpu_args[NUM_QPUS];
} ZeroHost;

/** @brief Initializes the zero host. */
ZeroHost* zero_host_init(AtomicEventRingBuffer *rb, VirtualMemoryContext *ctx) {
    ZeroHost *host = malloc(sizeof(ZeroHost));
    if (!host) {
        syslog(LOG_ERR, "ZeroHost: Failed to allocate");
        return NULL;
    }
    memset(host, 0, sizeof(ZeroHost));
    host->event_queue = rb;
    host->vmm_ctx = ctx;
    simple_lock_init(&host->lock);
    host->current_level = malloc(sizeof(LevelN));
    if (!host->current_level) {
        syslog(LOG_ERR, "ZeroHost: Failed to allocate current_level");
        simple_lock_destroy(&host->lock);
        free(host);
        return NULL;
    }
    memset(host->current_level, 0, sizeof(LevelN));
    host->current_level->scene = (Scene3D*)vmm_map_page(ctx, 0x1000, true, true);
    host->current_level->output_combined = malloc(sizeof(QPUOutput));
    if (!host->current_level->scene || !host->current_level->output_combined) {
        syslog(LOG_ERR, "ZeroHost: Failed to allocate scene or output");
        free(host->current_level->scene);
        free(host->current_level->output_combined);
        free(host->current_level);
        simple_lock_destroy(&host->lock);
        free(host);
        return NULL;
    }
    host->current_level->level_id = 0;
    host->current_level->next_level = NULL;
    host->current_level->scene->cycle = 0;
    host->current_level->scene->scene_id = 0;
    for (int i = 0; i < NUM_TRIANGLES; i++) {
        host->current_level->scene->triangles[i].v0 = (Vector3D){0.0f, 0.0f, 0.0f};
        host->current_level->scene->triangles[i].v1 = (Vector3D){1.0f, 0.0f, 0.0f};
        host->current_level->scene->triangles[i].v2 = (Vector3D){0.0f, 1.0f, 0.0f};
        host->current_level->scene->triangles[i].albedo = 0.5f;
        host->current_level->scene->triangles[i].texture_id = 0;
    }
    for (int i = 0; i < NUM_LIGHTS; i++) {
        host->current_level->scene->lights[i].pos = (Vector3D){(float)i, (float)i, 10.0f};
        host->current_level->scene->lights[i].intensity = 1.0f;
    }
    for (int i = 0; i < NUM_RAYS; i++) {
        host->current_level->scene->rays[i].origin = (Vector3D){0.0f, 0.0f, -5.0f};
        host->current_level->scene->rays[i].dir = (Vector3D){0.0f, 0.0f, 1.0f};
        host->current_level->scene->rays[i].intensity = 1.0f;
    }
    for (int i = 0; i < NUM_QPUS; i++) {
        host->qpu_ctxs[i] = qpu_init(NUM_QUBITS, (unsigned int)time(NULL) + i);
        host->output_buffers[i] = (QPUOutput*)vmm_map_page(ctx, 0x2000 + i, true, true);
        if (!host->qpu_ctxs[i] || !host->output_buffers[i]) {
            syslog(LOG_ERR, "ZeroHost: Failed to init QPU %d", i);
            for (int j = 0; j < i; j++) {
                qpu_free(host->qpu_ctxs[j]);
                free(host->output_buffers[j]);
            }
            free(host->current_level->scene);
            free(host->current_level->output_combined);
            free(host->current_level);
            simple_lock_destroy(&host->lock);
            free(host);
            return NULL;
        }
        host->qpu_threads[i] = 0;
        host->qpu_args[i] = (QPUArgs){host, i};
    }
    for (int i = 0; i < NUM_QPUS; i++) {
        host->current_level->scene->textures[i].width = IMAGE_WIDTH;
        host->current_level->scene->textures[i].height = IMAGE_HEIGHT;
        for (int j = 0; j < NUM_QUBITS; j++) {
            host->current_level->scene->textures[i].noise_params[j] = (float)xorshift32(&(unsigned int){(unsigned int)time(NULL)}) / UINT_MAX;
        }
    }
    syslog(LOG_INFO, "ZeroHost: Initialized with %d QPUs", NUM_QPUS);
    return host;
}

/** @brief Processes a batch of events. */
void zero_host_process_batch(ZeroHost *host) {
    simple_lock(&host->lock);
    Scene3D *scene = host->current_level->scene;
    if (host->batch_count == 0 || scene->cycle >= MAX_CYCLES) {
        syslog(LOG_INFO, "ZeroHost: Batch skipped (Cycle %d, count %d)", scene->cycle, host->batch_count);
        host->batch_count = 0;
        host->last_batch_time = 0;
        simple_unlock(&host->lock);
        return;
    }
    scene->scene_id = host->batch[0].scene_id + 1;
    scene->cycle++;
    float batch_brightness_sum = 0.0f;
    for (int i = 0; i < host->batch_count; i++) {
        batch_brightness_sum += host->batch[i].brightness;
    }
    float average_batch_brightness = batch_brightness_sum / host->batch_count;
    syslog(LOG_INFO, "ZeroHost: Processing batch (Cycle %d, avg brightness %.2f)", scene->cycle, average_batch_brightness);
    float sin_cache[NUM_TRIANGLES], cos_cache[NUM_TRIANGLES];
    for (int i = 0; i < NUM_TRIANGLES; i++) {
        sin_cache[i] = sinf(i * 0.1f + scene->cycle * 0.01f);
        cos_cache[i] = cosf(i * 0.1f + scene->cycle * 0.01f);
        scene->triangles[i].v0.x = sin_cache[i];
        scene->triangles[i].v0.y = cos_cache[i];
        scene->triangles[i].v0.z = sin_cache[i] * cos_cache[i] * average_batch_brightness * 2.0f +
                                   perlin_noise(i, scene->cycle, scene->textures[0].noise_params);
        scene->triangles[i].v1.x = scene->triangles[i].v0.x + 1.0f;
        scene->triangles[i].v1.y = scene->triangles[i].v0.y;
        scene->triangles[i].v1.z = scene->triangles[i].v0.z;
        scene->triangles[i].v2.x = scene->triangles[i].v0.x;
        scene->triangles[i].v2.y = scene->triangles[i].v0.y + 1.0f;
        scene->triangles[i].v2.z = scene->triangles[i].v0.z;
        scene->triangles[i].albedo = 0.5f + average_batch_brightness * 0.4f;
        scene->triangles[i].texture_id = i % NUM_QPUS;
    }
    for (int i = 0; i < NUM_LIGHTS; i++) {
        scene->lights[i].pos.x = (float)i / NUM_LIGHTS + sinf(scene->cycle * 0.02f);
        scene->lights[i].pos.y = 0.5f + cosf(scene->cycle * 0.03f);
        scene->lights[i].pos.z = 10.0f;
        scene->lights[i].intensity = (1.0f + sinf(scene->cycle * 2.0f)) * (1.0f + average_batch_brightness * 3.0f);
    }
    for (int i = 0; i < host->batch_count; i++) {
        int idx = host->batch[i].ray_idx;
        if (idx < NUM_RAYS) {
            scene->rays[idx].origin.x = (float)idx / NUM_RAYS + sinf(host->batch[i].brightness * 0.5f);
            scene->rays[idx].origin.y = (float)(idx * 2) / NUM_RAYS + cosf(host->batch[i].brightness * 0.5f);
            scene->rays[idx].origin.z = -5.0f;
            scene->rays[idx].dir = (Vector3D){0.0f, 0.0f, 1.0f};
            scene->rays[idx].intensity = 1.0f + host->batch[i].brightness;
        }
    }
    for (int i = 0; i < NUM_QPUS; i++) {
        if (pthread_create(&host->qpu_threads[i], NULL, qpu_thread, &host->qpu_args[i]) != 0) {
            syslog(LOG_ERR, "ZeroHost: Failed to create QPU thread %d: %s", i, strerror(errno));
        }
    }
    for (int i = 0; i < NUM_QPUS; i++) {
        if (host->qpu_threads[i] != 0) {
            pthread_join(host->qpu_threads[i], NULL);
            host->qpu_threads[i] = 0;
        }
    }
    QPUOutput *combined_output = host->current_level->output_combined;
    combined_output->derived_scene_id = scene->scene_id;
    for (int i = 0; i < NUM_TRIANGLES; i++) {
        float sum_albedo = 0.0f;
        for (int q = 0; q < NUM_QPUS; q++) {
            sum_albedo += host->output_buffers[q]->surface_albedo[i];
        }
        combined_output->surface_albedo[i] = sum_albedo / NUM_QPUS;
    }
    for (int i = 0; i < NUM_LIGHTS; i++) {
        float sum_intensity = 0.0f;
        for (int q = 0; q < NUM_QPUS; q++) {
            sum_intensity += host->output_buffers[q]->light_sources_intensity[i];
        }
        combined_output->light_sources_intensity[i] = sum_intensity / NUM_QPUS;
    }
    memcpy(combined_output->noise_seed_from_qpu, host->output_buffers[0]->noise_seed_from_qpu, NUM_QUBITS * sizeof(float));
    render_gpu(combined_output, scene, host->event_queue, scene->cycle);
    host->batch_count = 0;
    host->last_batch_time = 0;
    simple_unlock(&host->lock);
}

/** @brief QPU thread function. */
void* qpu_thread(void *arg) {
    QPUArgs *args = (QPUArgs*)arg;
    ZeroHost *host = args->host;
    int qpu_id = args->qpu_id;
    syslog(LOG_DEBUG, "QPU Thread %d: Started", qpu_id);
    QPUCircuit *circ = qpu_create_circuit(host->qpu_ctxs[qpu_id], NUM_QUBITS);
    if (!circ) {
        syslog(LOG_ERR, "QPU Thread %d: Failed to create circuit", qpu_id);
        return NULL;
    }
    qpu_execute(circ, host->qpu_ctxs[qpu_id], host->current_level->scene, host->output_buffers[qpu_id], qpu_id);
    qpu_free_circuit(circ);
    syslog(LOG_DEBUG, "QPU Thread %d: Finished", qpu_id);
    return NULL;
}

/** @brief Processes a single event. */
void zero_host_process_event(ZeroHost *host, Event *event) {
    if (!host || !event) return;
    if (host->current_level->scene->cycle >= MAX_CYCLES) return;
    simple_lock(&host->lock);
    host->batch[host->batch_count++] = *event;
    if (host->batch_count >= BATCH_SIZE || 
        (host->batch_count > 0 && usec_since(host->last_batch_time) > BATCH_TIMEOUT_US)) {
        simple_unlock(&host->lock);
        zero_host_process_batch(host);
    } else {
        if (host->batch_count == 1) host->last_batch_time = get_usec();
        simple_unlock(&host->lock);
    }
}

/** @brief Frees the zero host resources. */
void zero_host_free(ZeroHost *host) {
    if (!host) return;
    if (host->batch_count > 0 && host->current_level->scene->cycle < MAX_CYCLES) {
        syslog(LOG_INFO, "ZeroHost: Processing final batch");
        zero_host_process_batch(host);
    }
    for (int i = 0; i < NUM_QPUS; i++) {
        if (host->qpu_threads[i] != 0) {
            syslog(LOG_DEBUG, "ZeroHost: Joining QPU thread %d", i);
            pthread_join(host->qpu_threads[i], NULL);
        }
        qpu_free(host->qpu_ctxs[i]);
        free(host->output_buffers[i]);
    }
    if (host->current_level) {
        LevelN *level = host->current_level;
        while (level) {
            LevelN *next = level->next_level;
            free(level->scene);
            free(level->output_combined);
            free(level);
            level = next;
        }
    }
    simple_lock_destroy(&host->lock);
    free(host);
    syslog(LOG_INFO, "ZeroHost: Freed");
}

// --- Utilities ---
/** @brief Gets current time in microseconds. */
uint64_t get_usec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

/** @brief Calculates time elapsed since start. */
uint64_t usec_since(uint64_t start) {
    uint64_t now = get_usec();
    return now >= start ? now - start : 0;
}

// --- Worker Thread ---
/** @brief Generates initial events. */
void* worker_thread(void *arg) {
    WorkerArgs *args = (WorkerArgs*)arg;
    ZeroHost *host = args->host;
    int thread_idx = args->thread_idx;
    if (!host) {
        syslog(LOG_ERR, "Worker thread %d: Invalid host", thread_idx);
        return NULL;
    }
    unsigned int seed = (unsigned int)pthread_self() ^ (unsigned int)time(NULL) ^ thread_idx;
    int rays_per_worker = NUM_RAYS / NUM_THREADS;
    int start_ray = thread_idx * rays_per_worker;
    int end_ray = (thread_idx == NUM_THREADS - 1) ? NUM_RAYS : start_ray + rays_per_worker;
    for (int i = start_ray; i < end_ray; i++) {
        Event event = {
            .event_id = ((uint64_t)thread_idx << 32) | i,
            .thread_id = thread_idx,
            .ray_idx = i,
            .scene_id = 0,
            .brightness = (float)xorshift32(&seed) / UINT_MAX
        };
        if (enqueue_event(host->event_queue, &event) != 0) {
            syslog(LOG_WARNING, "Worker thread %d: Failed to enqueue event %lu", thread_idx, event.event_id);
        }
    }
    syslog(LOG_INFO, "Worker thread %d: Generated %d events", thread_idx, end_ray - start_ray);
    return NULL;
}

// --- Main ---
/** @brief Main entry point for the QPU 3D simulation. */
int main(void) {
    openlog("QPU3DSimV12", LOG_PID | LOG_CONS, LOG_USER);
    printf("--- Testing QPU 3D Simulation V12 ---\n");
    srand(time(NULL));
    if (vmm_init() != 0) {
        syslog(LOG_CRIT, "VMM initialization failed");
        closelog();
        return 1;
    }
    VirtualMemoryContext *ctx = vmm_allocate_process_context(0);
    if (!ctx) {
        syslog(LOG_CRIT, "Failed to allocate VMM context");
        closelog();
        return 1;
    }
    AtomicEventRingBuffer rb;
    ring_buffer_init(&rb);
    ZeroHost *host = zero_host_init(&rb, ctx);
    if (!host) {
        syslog(LOG_CRIT, "ZeroHost initialization failed");
        free(ctx);
        closelog();
        return 1;
    }
    pthread_t worker_threads[NUM_THREADS];
    WorkerArgs worker_args[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        worker_args[i] = (WorkerArgs){host, i};
        if (pthread_create(&worker_threads[i], NULL, worker_thread, &worker_args[i]) != 0) {
            syslog(LOG_ERR, "Main: Failed to create worker thread %d: %s", i, strerror(errno));
        }
    }
    Event event;
    int processed_events = 0;
    int expected_events = NUM_RAYS + (MAX_CYCLES - 1);
    int wait_count = 0;
    while (host->current_level->scene->cycle < MAX_CYCLES && processed_events < expected_events && wait_count < MAX_WAIT_COUNT) {
        if (dequeue_event(&rb, &event) == 0) {
            zero_host_process_event(host, &event);
            processed_events++;
            wait_count = 0;
        } else {
            simple_lock(&host->lock);
            if (host->batch_count > 0 && usec_since(host->last_batch_time) > BATCH_TIMEOUT_US) {
                simple_unlock(&host->lock);
                zero_host_process_batch(host);
            } else {
                simple_unlock(&host->lock);
                usleep(1000);
                wait_count++;
            }
        }
    }
    syslog(LOG_INFO, "Main: Processed %d/%d events (Cycle %d)", processed_events, expected_events, host->current_level->scene->cycle);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(worker_threads[i], NULL);
    }
    zero_host_free(host);
    free(ctx);
    closelog();
    printf("--- Simulation Complete ---\n");
    return 0;
}
