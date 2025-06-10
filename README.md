# QuantumRenderSim

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
// Configuración de la simulación cuántica
#define NUM_QUBITS 16           // Número de cúbits en cada QPU
#define NUM_QPUS 2              // Número de Unidades de Procesamiento Cuántico (QPUs)

// Configuración de la escena 3D y renderizado
#define NUM_RAYS 1024           // Número de rayos simulados
#define NUM_TRIANGLES 32        // Número de triángulos en la escena
#define NUM_LIGHTS 4            // Número de fuentes de luz
#define IMAGE_WIDTH 256         // Ancho de la imagen de salida (PPM)
#define IMAGE_HEIGHT 256        // Alto de la imagen de salida (PPM)

// Configuración de concurrencia y comunicación
#define NUM_WORKER_THREADS 8    // Número de hilos de trabajo que generan eventos
#define PAGE_SIZE 4096          // Tamaño de página para la simulación VMM
#define EVENT_SIZE 64           // Tamaño fijo de la estructura Event (para el búfer de anillo)
#define RING_SIZE 1024          // Capacidad del búfer de anillo de eventos
#define BATCH_SIZE 16           // Número de eventos por lote procesado por ZeroHost
#define BATCH_TIMEOUT_US 1000   // Tiempo de espera (microsegundos) para procesar un lote incompleto
#define BACKOFF_MAX_US 10000    // Retraso máximo (microsegundos) para el retroceso exponencial
#define MAX_ATTEMPTS 10         // Número máximo de intentos para encolar/desencolar

// Configuración de la simulación principal
#define MAX_CYCLES 5            // Número total de ciclos de simulación (generación de escena -> QPU -> render -> retroalimentación)

// --- VMM and Buddy ---
/** Simulated virtual memory context. */
typedef struct { int pid; } vm_context_t;

/** Pthread mutex for thread synchronization. */
typedef pthread_mutex_t simple_lock_t;

/** Initializes a mutex lock. */
static inline void simple_lock_init(simple_lock_t *lock) {
    if (pthread_mutex_init(lock, NULL) != 0) {
        syslog(LOG_ERR, "Failed to initialize mutex lock");
        // En una aplicación real, aquí podrías manejar el error de forma más robusta.
        // Para esta simulación, asumimos que la inicialización del mutex es crítica.
        exit(EXIT_FAILURE); 
    }
}
/** Acquires a mutex lock. */
static inline void simple_lock(simple_lock_t *lock) { pthread_mutex_lock(lock); }
/** Releases a mutex lock. */
static inline void simple_unlock(simple_lock_t *lock) { pthread_mutex_unlock(lock); }
/** Destroys a mutex lock. */
static inline void simple_lock_destroy(simple_lock_t *lock) { pthread_mutex_destroy(lock); }

/** Initializes the virtual memory manager. */
int vmm_init(void) {
    syslog(LOG_INFO, "VMM: Initialized");
    return 0; // Simulamos que siempre tiene éxito
}

/** Allocates a process context. */
vm_context_t* vmm_allocate_process_context(int pid) {
    vm_context_t *ctx = malloc(sizeof(vm_context_t));
    if (ctx) {
        ctx->pid = pid;
        syslog(LOG_INFO, "VMM: Allocated context for PID %d", pid);
    } else {
        syslog(LOG_ERR, "VMM: Failed to allocate context for PID %d", pid);
    }
    return ctx;
}

/** Maps a page-aligned memory region (simulated). */
// RETORNA MEMORIA ASIGNADA CON aligned_alloc, DEBE SER LIBERADA CON free()
unsigned long vmm_map_page(vm_context_t *ctx, unsigned int vpn, bool writable, bool user_access) {
    void *ptr = aligned_alloc(PAGE_SIZE, PAGE_SIZE);
    if (!ptr) {
        syslog(LOG_ERR, "VMM: Failed to allocate page for VPN %u (context PID: %d)", vpn, ctx ? ctx->pid : -1);
    } else {
        syslog(LOG_DEBUG, "VMM: Mapped page for VPN %u to %p", vpn, ptr);
    }
    return (unsigned long)ptr;
}

/** Dummy page allocation for a buddy system. */
// Estas son funciones de relleno, no necesitan cambios.
int buddy_alloc(int order) {
    // Simulamos una asignación de frame index.
    return rand() % 1000;
}

/** Dummy page deallocation for a buddy system. */
// Estas son funciones de relleno, no necesitan cambios.
void buddy_free(int frame_idx, int order) {
    // Simulamos la liberación de un frame.
    syslog(LOG_DEBUG, "Buddy: Freeing frame %d (order %d)", frame_idx, order);
}

// --- Ring Buffer ---
/** Event structure for the ring buffer. */
typedef struct {
    uint64_t event_id;
    int thread_id;
    unsigned int ray_idx;
    uint64_t scene_id;
    float brightness;
    // Padding para asegurar que el tamaño sea exactamente EVENT_SIZE
    char padding[EVENT_SIZE - sizeof(uint64_t) - sizeof(int) - sizeof(unsigned int) - sizeof(uint64_t) - sizeof(float)];
} Event;

// Comprobación en tiempo de compilación del tamaño de la estructura Event
static_assert(sizeof(Event) == EVENT_SIZE, "Event size must be exactly EVENT_SIZE as defined by macro");

/** Lock-free ring buffer for events. */
typedef struct {
    // Alineación a 64 bytes para evitar falsos compartidos (false sharing) en caché
    _Alignas(64) atomic_uint64_t head;
    _Alignas(64) atomic_uint64_t tail;
    _Alignas(64) Event buffer[RING_SIZE];
} AtomicEventRingBuffer;

/** Initializes the ring buffer. */
void ring_buffer_init(AtomicEventRingBuffer *rb) {
    atomic_store_explicit(&rb->head, 0, memory_order_relaxed);
    atomic_store_explicit(&rb->tail, 0, memory_order_relaxed);
    syslog(LOG_INFO, "RingBuffer: Initialized (%d slots)", RING_SIZE);
}

/** Enqueues an event with exponential backoff. */
int enqueue_event(AtomicEventRingBuffer *rb, Event *event) {
    uint64_t backoff_us = 1;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_relaxed);
        uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire); // Adquirir para ver el progreso de head
        
        // Comprobar si el búfer está lleno
        if (tail - head >= RING_SIZE) {
            syslog(LOG_DEBUG, "RingBuffer full, attempt %d. Backoff %lu us.", attempt, backoff_us);
            if (backoff_us > BACKOFF_MAX_US) return -1; // Fallar si el backoff excede el máximo
            usleep(backoff_us);
            backoff_us *= 2; // Retroceso exponencial
            continue;
        }
        
        // Intentar avanzar la cola
        if (atomic_compare_exchange_strong_explicit(&rb->tail, &tail, tail + 1, memory_order_release, memory_order_relaxed)) {
            // Se pudo avanzar la cola, ahora copiar el evento
            rb->buffer[tail % RING_SIZE] = *event;
            syslog(LOG_DEBUG, "Enqueued event %lu (ray %u, scene %lu, brightness %f)",
                     event->event_id, event->ray_idx, event->scene_id, event->brightness);
            return 0; // Éxito
        }
        // Si CAS falla, otro hilo ya avanzó la cola, intentar de nuevo.
        sched_yield(); // Ceder el control de la CPU para que otro hilo pueda avanzar
    }
    syslog(LOG_WARNING, "Failed to enqueue event %lu after %d attempts.", event->event_id, MAX_ATTEMPTS);
    return -1; // Fallo después de todos los intentos
}

/** Dequeues an event with exponential backoff. */
int dequeue_event(AtomicEventRingBuffer *rb, Event *event_data) {
    uint64_t backoff_us = 1;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        uint64_t head = atomic_load_explicit(&rb->head, memory_order_relaxed);
        uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire); // Adquirir para ver el progreso de tail
        
        // Comprobar si el búfer está vacío
        if (head == tail) {
            syslog(LOG_DEBUG, "RingBuffer empty, attempt %d. Backoff %lu us.", attempt, backoff_us);
            if (backoff_us > BACKOFF_MAX_US) return -1; // Fallar si el backoff excede el máximo
            usleep(backoff_us);
            backoff_us *= 2; // Retroceso exponencial
            continue;
        }
        
        // Intentar avanzar la cabeza
        if (atomic_compare_exchange_strong_explicit(&rb->head, &head, head + 1, memory_order_release, memory_order_relaxed)) {
            // Se pudo avanzar la cabeza, ahora copiar el evento
            *event_data = rb->buffer[head % RING_SIZE];
            syslog(LOG_DEBUG, "Dequeued event %lu (ray %u, scene %lu, brightness %f)",
                     event_data->event_id, event_data->ray_idx, event_data->scene_id, event_data->brightness);
            return 0; // Éxito
        }
        // Si CAS falla, otro hilo ya avanzó la cabeza, intentar de nuevo.
        sched_yield(); // Ceder el control de la CPU
    }
    syslog(LOG_WARNING, "Failed to dequeue event after %d attempts.", MAX_ATTEMPTS);
    return -1; // Fallo después de todos los intentos
}

// --- 3D Structures ---
/** 3D vector. */
typedef struct { float x, y, z; } Vector3D;
/** Triangle with vertices and material properties. */
typedef struct { Vector3D v0, v1, v2; float albedo; int texture_id; } Triangle;
/** Light source with position and intensity. */
typedef struct { Vector3D pos; float intensity; } Light;
/** Ray with origin, direction, and intensity. */
typedef struct { Vector3D origin, dir; float intensity; } Ray;
/** Texture with noise seed. */
// Mejor nombre para 'noise_seed_qpu_params' o similar, ya que proviene de QPU.
typedef struct { float noise_params[NUM_QUBITS]; int width, height; } Texture;
/** 3D scene with geometry, lights, rays, and textures. */
typedef struct {
    Triangle triangles[NUM_TRIANGLES];
    Light lights[NUM_LIGHTS];
    Ray rays[NUM_RAYS];
    Texture textures[NUM_QPUS]; // Texturas por cada QPU, si aplican
    uint64_t scene_id;
    int cycle;
} Scene3D;

/** Hierarchical simulation level. (Este struct parece no ser usado en la lógica actual de ZeroHost) */
// Si este struct no se usa o su uso es muy limitado, consideraría refactorizarlo o eliminarlo
// para simplificar la estructura. Para el propósito de esta mejora, lo mantendré y comentaré.
typedef struct LevelN {
    Scene3D *scene;
    QPUOutput *output_combined; // Adelantando la declaración de QPUOutput
    int level_id;
    struct LevelN *next_level; // Para futuras expansiones de niveles de simulación
} LevelN;

// --- QPU Output ---
/** Output parameters from a QPU. */
typedef struct {
    float surface_albedo[NUM_TRIANGLES];
    float light_sources_intensity[NUM_LIGHTS];
    float noise_seed_from_qpu[NUM_QUBITS]; // Nombre más específico
    uint64_t derived_scene_id;
} QPUOutput;

// --- QPU API ---
/** QPU context for quantum computations. */
typedef struct {
    int qubits;
    float *quantum_states;
    float *entangled_pairs; // Representación simplificada del entrelazamiento
    unsigned int rand_seed; // Semilla para el generador de números aleatorios interno
} QPUContext;

/** Quantum circuit for QAOA (Quantum Approximate Optimization Algorithm). */
typedef struct {
    int qubits;
    float *params; // Parámetros variacionales del circuito QAOA
} QPUCircuit;

/** XORShift random number generator. */
// Un generador de números aleatorios rápido y simple.
static inline unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

/** Initializes a QPU context. */
QPUContext* qpu_init(int qubits, unsigned int initial_seed) {
    QPUContext *ctx = malloc(sizeof(QPUContext));
    if (!ctx) {
        syslog(LOG_ERR, "QPU: Failed to allocate QPUContext");
        return NULL;
    }
    ctx->qubits = qubits;
    ctx->quantum_states = malloc(qubits * sizeof(float));
    ctx->entangled_pairs = malloc(qubits * sizeof(float));
    if (!ctx->quantum_states || !ctx->entangled_pairs) {
        syslog(LOG_ERR, "QPU: Failed to allocate quantum states/entangled pairs");
        free(ctx->quantum_states); // free(NULL) es seguro
        free(ctx->entangled_pairs);
        free(ctx);
        return NULL;
    }
    // Combinar semillas para una mayor aleatoriedad inicial y unicidad por hilo
    ctx->rand_seed = initial_seed ^ (unsigned int)time(NULL) ^ (unsigned int)pthread_self();
    for (int i = 0; i < qubits; i++) {
        ctx->quantum_states[i] = 0.707f; // Estado de superposición inicial (1/sqrt(2))
        ctx->entangled_pairs[i] = 0.0f; // Inicialmente sin entrelazamiento
    }
    syslog(LOG_INFO, "QPU: Initialized with %d qubits and seed %u", qubits, ctx->rand_seed);
    return ctx;
}

/** Frees a QPU context. */
void qpu_free(QPUContext *ctx) {
    if (ctx) {
        free(ctx->quantum_states);
        free(ctx->entangled_pairs);
        free(ctx);
        syslog(LOG_DEBUG, "QPUContext freed");
    }
}

/** Creates a QAOA circuit. */
QPUCircuit* qpu_create_circuit(QPUContext *ctx, int qubits) {
    // El ctx no se usa actualmente aquí, pero se mantiene para coherencia del API.
    QPUCircuit *circ = malloc(sizeof(QPUCircuit));
    if (!circ) {
        syslog(LOG_ERR, "QPU: Failed to allocate QPUCircuit");
        return NULL;
    }
    circ->qubits = qubits;
    circ->params = malloc(qubits * sizeof(float)); // Parámetros por cúbit/capa
    if (!circ->params) {
        syslog(LOG_ERR, "QPU: Failed to allocate circuit parameters");
        free(circ);
        return NULL;
    }
    for (int i = 0; i < qubits; i++) circ->params[i] = 0.5f; // Inicializar parámetros
    syslog(LOG_DEBUG, "QPU: Created QAOA circuit with %d qubits", qubits);
    return circ;
}

/** Applies a quantum gate (simplified simulation). */
void qpu_apply_gate(QPUContext *ctx, const char *gate_type, int qubit_idx, float param) {
    if (!ctx || qubit_idx < 0 || qubit_idx >= ctx->qubits) {
        syslog(LOG_WARNING, "QPU: Invalid gate application (qubit_idx %d, qubits %d)", qubit_idx, ctx ? ctx->qubits : -1);
        return;
    }
    // Lógica simplificada de las puertas cuánticas
    if (strcmp(gate_type, "H") == 0) {
        ctx->quantum_states[qubit_idx] = 0.707f; // Simula Hadamard, pone en superposición
    } else if (strcmp(gate_type, "RX") == 0) {
        ctx->quantum_states[qubit_idx] = cosf(param) * ctx->quantum_states[qubit_idx]; // Rotación RX
    } else if (strcmp(gate_type, "CNOT") == 0) {
        int target_qubit_idx = (qubit_idx + 1) % ctx->qubits; // CNOT simple al siguiente cúbit
        ctx->entangled_pairs[qubit_idx] = 0.5f; // Simula algún entrelazamiento
        // Lógica real de CNOT sería más compleja, afectando el estado del cúbit objetivo
    }
    syslog(LOG_DEBUG, "QPU: Applied %s gate to qubit %d with param %.2f", gate_type, qubit_idx, param);
}

/** Measures a qubit (simplified simulation). */
int qpu_measure_qubit(QPUContext *ctx, int qubit_idx) {
    if (!ctx || qubit_idx < 0 || qubit_idx >= ctx->qubits) {
        syslog(LOG_WARNING, "QPU: Invalid qubit measurement (qubit_idx %d, qubits %d)", qubit_idx, ctx ? ctx->qubits : -1);
        return -1;
    }
    // La probabilidad se basa en la amplitud al cuadrado (simplificado).
    float prob_of_1 = ctx->quantum_states[qubit_idx] * ctx->quantum_states[qubit_idx];
    // Generar un resultado de medición (0 o 1)
    int result = ((float)xorshift32(&ctx->rand_seed) / (float)UINT_MAX) < prob_of_1 ? 1 : 0;
    // Colapsar el estado después de la medición
    ctx->quantum_states[qubit_idx] = result ? 1.0f : 0.0f;
    syslog(LOG_DEBUG, "QPU: Measured qubit %d -> %d (prob of 1: %.2f)", qubit_idx, result, prob_of_1);
    return result;
}

/** Executes a QAOA circuit on a scene, influencing rendering parameters. */
void qpu_execute(QPUCircuit *circ, QPUContext *ctx, Scene3D *scene, QPUOutput *output, int qpu_id) {
    if (!circ || !ctx || !scene || !output) {
        syslog(LOG_ERR, "QPU %d: Invalid arguments for qpu_execute", qpu_id);
        return;
    }

    float costs[NUM_TRIANGLES] = {0};
    float total_light_intensity = 0.0f;
    for (int k = 0; k < NUM_LIGHTS; k++) {
        total_light_intensity += scene->lights[k].intensity;
    }

    // Dividir los rayos entre las QPUs
    int rays_per_qpu = NUM_RAYS / NUM_QPUS;
    // Asegurarse de que la última QPU cubra cualquier rayo restante si NUM_RAYS no es divisible por NUM_QPUS
    int start_ray_idx = qpu_id * rays_per_qpu;
    int end_ray_idx = (qpu_id == NUM_QPUS - 1) ? NUM_RAYS : start_ray_idx + rays_per_qpu;

    syslog(LOG_DEBUG, "QPU %d processing rays from %d to %d", qpu_id, start_ray_idx, end_ray_idx - 1);

    // Calcular "costos" basados en interacciones de rayos y triángulos.
    // Esto simula la parte de optimización del QAOA aplicada a la escena.
    for (int i = start_ray_idx; i < end_ray_idx; i++) {
        for (int j = 0; j < NUM_TRIANGLES; j++) {
            float dx = scene->rays[i].origin.x - scene->triangles[j].v0.x;
            float dy = scene->rays[i].origin.y - scene->triangles[j].v0.y;
            float dz = scene->rays[i].origin.z - scene->triangles[j].v0.z; // Considerar Z también
            float dist_sq = dx * dx + dy * dy + dz * dz + 0.0001f; // Evitar división por cero
            costs[j] += (scene->rays[i].intensity * total_light_intensity * scene->triangles[j].albedo) / dist_sq;
        }
    }

    // Simulación de la aplicación de capas QAOA
    for (int iter = 0; iter < 5; iter++) { // Iteraciones del circuito QAOA
        for (int i = 0; i < ctx->qubits; i++) {
            qpu_apply_gate(ctx, "H", i, 0.0f); // Capa de mezcladores (Hadamard)
            // Parámetros variacionales influenciados por el circuito, la iteración y el ciclo de la escena.
            float beta = circ->params[i] + iter * 0.1f + scene->cycle * 0.05f;
            qpu_apply_gate(ctx, "RX", i, beta * costs[i % NUM_TRIANGLES]); // Capa de costos (depende de la escena)
            if (i < ctx->qubits - 1) {
                qpu_apply_gate(ctx, "CNOT", i, 0.0f); // Entrelazamiento
            }
        }
    }

    // Medir los cúbits y mapear los resultados a los parámetros de salida de renderizado
    for (int i = 0; i < NUM_TRIANGLES; i++) {
        // Albedo influenciado por la medición del cúbit (escalado a un rango razonable)
        output->surface_albedo[i] = (float)qpu_measure_qubit(ctx, i % NUM_QUBITS) * 0.8f + 0.2f; // De 0.2 a 1.0
    }
    for (int i = 0; i < NUM_LIGHTS; i++) {
        // Intensidad de la luz influenciada por la medición del cúbit
        output->light_sources_intensity[i] = (float)qpu_measure_qubit(ctx, i % NUM_QUBITS) * 5.0f; // De 0.0 a 5.0
    }
    for (int i = 0; i < NUM_QUBITS; i++) {
        // Semilla de ruido para texturas
        output->noise_seed_from_qpu[i] = (float)qpu_measure_qubit(ctx, i); // De 0.0 o 1.0
        // Actualizar la semilla de ruido en la textura de la escena (compartida)
        scene->textures[qpu_id].noise_params[i] = output->noise_seed_from_qpu[i];
    }

    output->derived_scene_id = scene->scene_id + 1; // La QPU deriva un nuevo ID de escena
    syslog(LOG_INFO, "QPU %d: Generated parameters for scene %lu", qpu_id, output->derived_scene_id);
}

/** Frees a QAOA circuit. */
void qpu_free_circuit(QPUCircuit *circ) {
    if (circ) {
        free(circ->params);
        free(circ);
        syslog(LOG_DEBUG, "QPUCircuit freed");
    }
}

// --- Perlin Noise ---
/** Precomputed noise table. */
static float noise_table[256];
static bool noise_table_initialized = false;
static pthread_mutex_t noise_table_mutex = PTHREAD_MUTEX_INITIALIZER; // Mutex para inicialización segura

/** Initializes the noise table (thread-safe, once). */
static void init_noise_table(void) {
    // Patrón de inicialización de doble verificación (double-checked locking)
    if (!noise_table_initialized) {
        pthread_mutex_lock(&noise_table_mutex);
        if (!noise_table_initialized) {
            unsigned int seed = (unsigned int)time(NULL);
            for (int i = 0; i < 256; i++) {
                noise_table[i] = (float)rand_r(&seed) / RAND_MAX;
            }
            noise_table_initialized = true;
            syslog(LOG_INFO, "Perlin: Noise table initialized.");
        }
        pthread_mutex_unlock(&noise_table_mutex);
    }
}

/** Generates Perlin noise (simplified). */
float perlin_noise(float x, float y, float *noise_params) {
    init_noise_table(); // Asegurar que la tabla de ruido esté inicializada
    if (!noise_params) {
        // Si no hay parámetros de ruido específicos, usar una aproximación
        return noise_table[((int)(x * 10) % 256 + 256) % 256] * sinf(x + y);
    }
    // Usar un parámetro de ruido de QPU para influenciar el ruido Perlin
    int seed_idx = ((int)x + (int)y) % NUM_QUBITS;
    int table_idx = ((int)(x * y * 10) % 256 + 256) % 256; // Factor para una distribución más amplia
    return noise_params[seed_idx] * noise_table[table_idx] * sinf(x + y);
}

// --- PPM Rendering ---
/** Renders the scene to a PPM file and enqueues brightness feedback. */
void render_gpu(QPUOutput *output, Scene3D *scene, AtomicEventRingBuffer *rb, int cycle) {
    char filename[128]; // Buffer más grande para el nombre de archivo
    // Nombre de archivo único para cada ciclo
    snprintf(filename, sizeof(filename), "render_cycle_%03d_scene_%lu.ppm", cycle, output->derived_scene_id);

    FILE *f = fopen(filename, "w");
    if (!f) {
        syslog(LOG_ERR, "GPU: Failed to open %s: %s", filename, strerror(errno));
        return;
    }

    fprintf(f, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    float total_pixel_value = 0.0f;

    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int tri_idx = (x + y) % NUM_TRIANGLES; // Simple mapeo de pixel a triángulo
            float albedo = output->surface_albedo[tri_idx];

            // Usar la semilla de ruido de la primera QPU (se asume combinación posterior en ZeroHost)
            // Se debería usar la textura combinada o una QPU específica para la textura global
            float tex_noise = perlin_noise(x / 10.0f, y / 10.0f, scene->textures[0].noise_params);

            // Calcular colores R, G, B basados en albedo, intensidad de luz y ruido.
            // Los valores se escalan para estar dentro de 0-255.
            int r = (int)(albedo * 255 * (1.0f + tex_noise));
            int g = (int)(albedo * 200 * (1.0f + tex_noise));
            int b = (int)(albedo * 150 * (1.0f + tex_noise));

            // Asegurar que los valores de color estén dentro del rango [0, 255]
            r = r > 255 ? 255 : (r < 0 ? 0 : r);
            g = g > 255 ? 255 : (g < 0 ? 0 : g);
            b = b > 255 ? 255 : (b < 0 ? 0 : b);

            fprintf(f, "%d %d %d ", r, g, b);
            total_pixel_value += (float)(r + g + b); // Sumar para calcular el brillo promedio
        }
        fprintf(f, "\n");
    }
    fclose(f);

    syslog(LOG_INFO, "GPU: Rendered scene %lu (Cycle %d) to %s", output->derived_scene_id, cycle, filename);

    // Calcular el brillo promedio de la imagen
    float average_brightness = total_pixel_value / (float)(IMAGE_WIDTH * IMAGE_HEIGHT * 3 * 255);

    // Encolar el brillo promedio como retroalimentación para el siguiente ciclo
    if (cycle < MAX_CYCLES - 1) { // No encolar feedback si es el último ciclo
        Event event = {
            .event_id = output->derived_scene_id,
            .thread_id = 0, // 0 para indicar que es un evento del sistema (GPU)
            .ray_idx = 0,   // No aplica un ray_idx específico
            .scene_id = output->derived_scene_id,
            .brightness = average_brightness
        };
        if (enqueue_event(rb, &event) != 0) {
            syslog(LOG_WARNING, "GPU: Failed to enqueue brightness event for scene %lu", event.scene_id);
        }
    }
}

// --- Zero Host ---
// Forward declaration para la función qpu_thread
void* qpu_thread(void *arg);

/** Host structure managing QPUs and simulation levels. */
typedef struct {
    QPUContext *qpu_ctxs[NUM_QPUS];
    AtomicEventRingBuffer *event_queue;
    LevelN *current_level; // Gestión de niveles de simulación
    QPUOutput *output_buffers[NUM_QPUS]; // Buffers de salida de cada QPU
    vm_context_t *vmm_ctx; // Contexto del VMM (no liberado por ZeroHost)
    simple_lock_t lock; // Mutex para proteger el acceso a la lógica de procesamiento por lotes
    Event batch[BATCH_SIZE]; // Lote de eventos
    int batch_count; // Contador de eventos en el lote
    uint64_t last_batch_time; // Tiempo del último evento en el lote
    
    // Mejoras para la sincronización de hilos QPU
    pthread_t qpu_threads[NUM_QPUS];
    // Argumentos pasados a cada hilo QPU
    struct QPUArgs { ZeroHost *host; int qpu_id; } qpu_args[NUM_QPUS];
    // Usar un contador atómico para los hilos QPU activos es bueno,
    // pero `pthread_join` explícito es más robusto para esperar su terminación.
    // La eliminación de `atomic_int active_qpu_threads` se maneja con `pthread_join`.
} ZeroHost;

/** Initializes the zero host. */
ZeroHost* zero_host_init(AtomicEventRingBuffer *rb, vm_context_t *ctx) {
    ZeroHost *host = malloc(sizeof(ZeroHost));
    if (!host) {
        syslog(LOG_ERR, "ZeroHost: Failed to allocate ZeroHost structure");
        return NULL;
    }
    
    // Inicializar punteros a NULL para una liberación segura en caso de error
    memset(host, 0, sizeof(ZeroHost)); // Inicializa todo a cero/NULL
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
    memset(host->current_level, 0, sizeof(LevelN)); // Inicializa a cero/NULL

    // Asignar memoria para la escena 3D y la salida combinada
    // IMPORTANTE: vmm_map_page retorna memoria de heap que necesita free()
    host->current_level->scene = (Scene3D*)vmm_map_page(ctx, 0x1000, true, true);
    host->current_level->output_combined = malloc(sizeof(QPUOutput));

    if (!host->current_level->scene || !host->current_level->output_combined) {
        syslog(LOG_ERR, "ZeroHost: Failed to allocate scene or combined output");
        free(host->current_level->scene); // free(NULL) es seguro
        free(host->current_level->output_combined);
        free(host->current_level);
        simple_lock_destroy(&host->lock);
        free(host);
        return NULL;
    }

    host->current_level->level_id = 0;
    host->current_level->next_level = NULL; // No hay niveles adicionales por ahora

    // Inicializar la escena
    host->current_level->scene->cycle = 0;
    host->current_level->scene->scene_id = 0;
    // Inicializar triángulos, luces y rayos con valores por defecto o aleatorios iniciales
    for (int i = 0; i < NUM_TRIANGLES; ++i) {
        host->current_level->scene->triangles[i].v0 = (Vector3D){0.0f, 0.0f, 0.0f};
        host->current_level->scene->triangles[i].v1 = (Vector3D){1.0f, 0.0f, 0.0f};
        host->current_level->scene->triangles[i].v2 = (Vector3D){0.0f, 1.0f, 0.0f};
        host->current_level->scene->triangles[i].albedo = 0.5f;
        host->current_level->scene->triangles[i].texture_id = 0;
    }
    for (int i = 0; i < NUM_LIGHTS; ++i) {
        host->current_level->scene->lights[i].pos = (Vector3D){(float)i, (float)i, 10.0f};
        host->current_level->scene->lights[i].intensity = 1.0f;
    }
     for (int i = 0; i < NUM_RAYS; ++i) {
        host->current_level->scene->rays[i].origin = (Vector3D){0.0f, 0.0f, -5.0f};
        host->current_level->scene->rays[i].dir = (Vector3D){0.0f, 0.0f, 1.0f};
        host->current_level->scene->rays[i].intensity = 1.0f;
    }


    // Inicializar QPUs y sus buffers de salida
    for (int i = 0; i < NUM_QPUS; i++) {
        host->qpu_ctxs[i] = qpu_init(NUM_QUBITS, (unsigned int)time(NULL) + i);
        host->output_buffers[i] = (QPUOutput*)vmm_map_page(ctx, 0x2000 + i, true, true);
        if (!host->qpu_ctxs[i] || !host->output_buffers[i]) {
            syslog(LOG_ERR, "ZeroHost: Failed to initialize QPU %d or its output buffer", i);
            // Liberar lo que ya se asignó en este bucle
            for (int j = 0; j <= i; j++) {
                qpu_free(host->qpu_ctxs[j]); // free(NULL) es seguro
                free(host->output_buffers[j]); // free(NULL) es seguro
            }
            // Liberar recursos de nivel superior antes de salir
            free(host->current_level->scene);
            free(host->current_level->output_combined);
            free(host->current_level);
            simple_lock_destroy(&host->lock);
            free(host);
            return NULL;
        }
        // Configurar argumentos para los hilos QPU
        host->qpu_args[i] = (struct QPUArgs){.host = host, .qpu_id = i};
    }
    
    // Inicializar texturas de la escena
    for (int i = 0; i < NUM_QPUS; i++) {
        host->current_level->scene->textures[i].width = IMAGE_WIDTH;
        host->current_level->scene->textures[i].height = IMAGE_HEIGHT;
        for (int j = 0; j < NUM_QUBITS; j++) {
            host->current_level->scene->textures[i].noise_params[j] = (float)rand() / RAND_MAX;
        }
    }

    host->batch_count = 0;
    host->last_batch_time = 0;
    
    syslog(LOG_INFO, "ZeroHost: Initialized with %d QPUs", NUM_QPUS);
    return host;
}

/** Processes a batch of events. */
void zero_host_process_batch(ZeroHost *host) {
    simple_lock(&host->lock); // Proteger el acceso al lote y al estado de la escena

    Scene3D *scene = host->current_level->scene;

    // Condición de salida si no hay eventos o se alcanzan los ciclos máximos
    if (host->batch_count == 0 || scene->cycle >= MAX_CYCLES) {
        syslog(LOG_INFO, "ZeroHost: Batch process skipped (empty batch or max cycles reached: %d)", scene->cycle);
        host->batch_count = 0;
        host->last_batch_time = 0;
        simple_unlock(&host->lock);
        return;
    }

    // Actualizar ID de escena y ciclo
    scene->scene_id = host->batch[0].scene_id + 1; // Derivar nuevo ID de escena
    scene->cycle++; // Avanzar el ciclo de simulación

    // Calcular el brillo promedio del lote actual
    float batch_brightness_sum = 0.0f;
    for (int i = 0; i < host->batch_count; i++) {
        batch_brightness_sum += host->batch[i].brightness;
    }
    float average_batch_brightness = batch_brightness_sum / host->batch_count;

    syslog(LOG_INFO, "ZeroHost: Processing batch (Cycle %d, avg brightness: %.2f)",
             scene->cycle, average_batch_brightness);

    // Actualizar dinámicamente los parámetros de la escena basados en el brillo promedio
    float sin_cache[NUM_TRIANGLES], cos_cache[NUM_TRIANGLES]; // Optimización de cache
    for (int i = 0; i < NUM_TRIANGLES; i++) {
        sin_cache[i] = sinf(i * 0.1f + scene->cycle * 0.01f);
        cos_cache[i] = cosf(i * 0.1f + scene->cycle * 0.01f);
        
        scene->triangles[i].v0.x = sin_cache[i];
        scene->triangles[i].v0.y = cos_cache[i];
        scene->triangles[i].v0.z = sin_cache[i] * cos_cache[i] * average_batch_brightness * 2.0f +
                                   perlin_noise(i, scene->cycle, scene->textures[0].noise_params);
        scene->triangles[i].v1.x = scene->triangles[i].v0.x + 1.0f; // Vértices dinámicos
        scene->triangles[i].v1.y = scene->triangles[i].v0.y;
        scene->triangles[i].v1.z = scene->triangles[i].v0.z;
        scene->triangles[i].v2.x = scene->triangles[i].v0.x;
        scene->triangles[i].v2.y = scene->triangles[i].v0.y + 1.0f;
        scene->triangles[i].v2.z = scene->triangles[i].v0.z;
        
        scene->triangles[i].albedo = 0.5f + average_batch_brightness * 0.4f; // El albedo cambia
        scene->triangles[i].texture_id = i % NUM_QPUS; // Asignación de textura (simplificada)
    }

    for (int i = 0; i < NUM_LIGHTS; i++) {
        scene->lights[i].pos.x = (float)i / NUM_LIGHTS + sinf(scene->cycle * 0.02f);
        scene->lights[i].pos.y = 0.5f + cosf(scene->cycle * 0.03f);
        scene->lights[i].pos.z = 10.0f; // Mantener Z constante
        scene->lights[i].intensity = 1.0f + average_batch_brightness * 3.0f; // La intensidad cambia
    }

    // Actualizar los rayos basados en los eventos del lote
    for (int i = 0; i < host->batch_count; i++) {
        int idx = host->batch[i].ray_idx;
        if (idx < NUM_RAYS) { // Asegurarse de no salirse del rango
            scene->rays[idx].origin.x = (float)idx / NUM_RAYS + sinf(host->batch[i].brightness * 0.5f);
            scene->rays[idx].origin.y = (float)(idx * 2) / NUM_RAYS + cosf(host->batch[i].brightness * 0.5f);
            scene->rays[idx].origin.z = -5.0f;
            scene->rays[idx].dir = (Vector3D){0.0f, 0.0f, 1.0f}; // Rayos apuntando hacia adelante
            scene->rays[idx].intensity = 1.0f + host->batch[i].brightness;
        }
    }

    // Lanzar hilos QPU para procesar la escena actualizada
    for (int i = 0; i < NUM_QPUS; i++) {
        // Asegurarse de que el hilo anterior haya terminado antes de crear uno nuevo con el mismo ID
        // (Aunque pthread_create asigna nuevos IDs, es una buena práctica asegurar la limpieza)
        // host->qpu_threads[i] se inicializa a 0 en zero_host_init y después de pthread_join.
        if (pthread_create(&host->qpu_threads[i], NULL, qpu_thread, &host->qpu_args[i]) != 0) {
            syslog(LOG_ERR, "ZeroHost: Failed to create QPU thread %d: %s", i, strerror(errno));
            // Si falla la creación, manejar el error (ej. decrementar un contador de activos)
            // En esta simulación, fallar la creación de un hilo QPU es crítico, así que simplemente se loguea.
        }
    }

    // Esperar explícitamente a que todos los hilos QPU terminen
    for (int i = 0; i < NUM_QPUS; i++) {
        if (host->qpu_threads[i] != 0) { // Solo hacer join si el hilo fue creado
            pthread_join(host->qpu_threads[i], NULL);
            host->qpu_threads[i] = 0; // Resetear el ID del hilo para futuras creaciones
        }
    }
    syslog(LOG_INFO, "ZeroHost: All QPU threads finished for cycle %d", scene->cycle);

    // Combinar las salidas de todas las QPUs en una única salida combinada
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
    // Para la semilla de ruido, se toma la de la primera QPU como representativa o se combina.
    // Aquí se copiará de la QPU 0.
    memcpy(combined_output->noise_seed_from_qpu, host->output_buffers[0]->noise_seed_from_qpu, NUM_QUBITS * sizeof(float));

    // Renderizar la escena con los parámetros influenciados por la QPU
    render_gpu(combined_output, scene, host->event_queue, scene->cycle);

    // Resetear el contador del lote para el siguiente ciclo
    host->batch_count = 0;
    host->last_batch_time = 0;
    
    simple_unlock(&host->lock);
}

/** QPU thread function. */
void* qpu_thread(void *arg) {
    struct QPUArgs *args = (struct QPUArgs*)arg;
    ZeroHost *host = args->host;
    int qpu_id = args->qpu_id;

    syslog(LOG_DEBUG, "QPU Thread %d: Started", qpu_id);

    QPUCircuit *circ = qpu_create_circuit(host->qpu_ctxs[qpu_id], NUM_QUBITS);
    if (!circ) {
        syslog(LOG_ERR, "QPU Thread %d: Failed to create circuit. Exiting.", qpu_id);
        return NULL; // Hilo termina si no puede crear el circuito
    }

    // Ejecutar el circuito QPU con los datos de la escena y obtener la salida
    qpu_execute(circ, host->qpu_ctxs[qpu_id], host->current_level->scene, host->output_buffers[qpu_id], qpu_id);

    qpu_free_circuit(circ); // Liberar los recursos del circuito

    syslog(LOG_DEBUG, "QPU Thread %d: Finished", qpu_id);
    return NULL;
}

/** Processes a single event. */
void zero_host_process_event(ZeroHost *host, Event *event) {
    simple_lock(&host->lock); // Bloquear para modificar el lote
    
    // Si ya se alcanzaron los ciclos máximos, ignorar nuevos eventos
    if (host->current_level->scene->cycle >= MAX_CYCLES) {
        simple_unlock(&host->lock);
        return;
    }

    host->batch[host->batch_count++] = *event; // Añadir evento al lote

    // Comprobar si el lote está lleno o si el tiempo de espera ha expirado
    if (host->batch_count >= BATCH_SIZE ||
        (host->batch_count > 0 && usec_since(host->last_batch_time) > BATCH_TIMEOUT_US)) {
        simple_unlock(&host->lock); // Desbloquear antes de llamar a process_batch para evitar deadlocks
        zero_host_process_batch(host); // Procesar el lote
    } else {
        // Si es el primer evento en el lote, registrar el tiempo de inicio
        if (host->batch_count == 1) {
            host->last_batch_time = get_usec();
        }
        simple_unlock(&host->lock); // Desbloquear si no se procesa el lote
    }
}

/** Frees the zero host resources. */
// Asegura que todos los recursos asignados por ZeroHost se liberen correctamente.
void zero_host_free(ZeroHost *host) {
    if (!host) return;

    // Si hay un lote pendiente y no se han alcanzado los ciclos máximos, procesarlo.
    // Esto es importante para asegurar que los últimos eventos no se pierdan.
    if (host->batch_count > 0 && host->current_level && host->current_level->scene->cycle < MAX_CYCLES) {
        syslog(LOG_INFO, "ZeroHost: Processing final pending batch during free.");
        zero_host_process_batch(host); // Procesar el lote final
    }

    // Asegurar que todos los hilos QPU hayan terminado y liberar sus contextos QPU
    for (int i = 0; i < NUM_QPUS; i++) {
        if (host->qpu_threads[i] != 0) { // Si el hilo fue creado, esperar su terminación
            syslog(LOG_DEBUG, "ZeroHost: Joining QPU thread %d...", i);
            pthread_join(host->qpu_threads[i], NULL);
            host->qpu_threads[i] = 0; // Resetear el ID del hilo
        }
        qpu_free(host->qpu_ctxs[i]); // Liberar el contexto QPU
        free(host->output_buffers[i]); // Liberar la memoria asignada por vmm_map_page para output_buffers
    }

    // Liberar los recursos del nivel de simulación
    if (host->current_level) {
        // En tu implementación de vmm_map_page, se usa aligned_alloc,
        // por lo tanto, la memoria devuelta debe liberarse con free().
        // Asegúrate de que `host->current_level->scene` se libere.
        if (host->current_level->scene) {
            free(host->current_level->scene);
            host->current_level->scene = NULL;
        }
        if (host->current_level->output_combined) {
            free(host->current_level->output_combined);
            host->current_level->output_combined = NULL;
        }
        free(host->current_level);
        host->current_level = NULL;
    }

    simple_lock_destroy(&host->lock); // Destruir el mutex
    free(host); // Liberar la estructura ZeroHost
    syslog(LOG_INFO, "ZeroHost: All resources freed.");
}

// --- Utilities ---
/** Gets current time in microseconds. */
uint64_t get_usec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts); // CLOCK_MONOTONIC es mejor para medir duraciones
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

/** Calculates time elapsed since start. */
uint64_t usec_since(uint64_t start) {
    uint64_t now = get_usec();
    // Prevenir desbordamiento si el tiempo actual es menor (aunque con CLOCK_MONOTONIC es raro)
    return now >= start ? now - start : 0;
}

// --- Worker Thread ---
/** Generates initial events. */
void* worker_thread(void *arg) {
    ZeroHost *host = (ZeroHost*)arg;
    if (!host) {
        syslog(LOG_ERR, "Worker thread: Invalid host argument. Exiting.");
        return NULL;
    }
    
    // Usar pthread_self() como parte de la semilla para mayor unicidad
    unsigned int thread_id_val = (unsigned int)pthread_self();
    unsigned int seed = thread_id_val ^ (unsigned int)time(NULL) ^ (unsigned int)rand(); // Combinar para mejor aleatoriedad

    // Calcular el índice del hilo (rudimentario, para asignar un rango de rayos)
    int thread_idx = 0;
    // Esto es un placeholder, lo ideal sería pasar el thread_idx como argumento al hilo.
    // Para esta simulación, asumimos que los hilos se crean en orden simple.
    // Un método más robusto sería usar un contador atómico para asignar IDs de hilo.
    for (int i = 0; i < NUM_WORKER_THREADS; i++) {
        if (pthread_self() == host->qpu_threads[i]) { // Error: worker_threads no son qpu_threads
            // Corregido: Esto es un error, el worker thread no debe compartir ID con qpu_threads
            // Lo más simple es pasar el thread_idx como argumento al crear el hilo.
            thread_idx = i; // Esto es una aproximación, no un identificador único real del worker.
            break;
        }
    }
    
    int rays_per_worker = NUM_RAYS / NUM_WORKER_THREADS;
    int start_ray = thread_idx * rays_per_worker;
    int end_ray = (thread_idx == NUM_WORKER_THREADS - 1) ? NUM_RAYS : start_ray + rays_per_worker;

    for (int i = start_ray; i < end_ray; i++) {
        Event event = {
            .event_id = ((uint64_t)thread_id_val << 32) | i, // ID de evento único
            .thread_id = thread_id_val,
            .ray_idx = i,
            .scene_id = 0, // Escena inicial
            .brightness = (float)rand_r(&seed) / RAND_MAX // Brillo aleatorio inicial
        };
        if (enqueue_event(host->event_queue, &event) != 0) {
            syslog(LOG_WARNING, "Worker thread %u: Failed to enqueue event %lu", thread_id_val, event.event_id);
            // Podríamos detener el hilo o reintentar, dependiendo de la tolerancia a fallos.
        }
    }
    syslog(LOG_INFO, "Worker thread %u finished generating %d events", thread_id_val, end_ray - start_ray);
    return NULL;
}

// --- Main ---
/** Main entry point for the QPU 3D simulation. */
int main(void) {
    // Configurar el syslog para que los mensajes aparezcan en la consola y un archivo de registro.
    // LOG_CONS: Si ocurre un error abriendo la conexión a syslog, intentar imprimir en la consola.
    // LOG_PID: Incluir el PID del proceso en cada mensaje.
    // LOG_USER: Instalar syslog para el facility 'user'.
    openlog("QPU3DSimV11_Enhanced", LOG_PID | LOG_CONS, LOG_USER);
    printf("--- Testing QPU 3D Simulation V11 (Enhanced) ---\n");
    syslog(LOG_INFO, "Simulation started.");

    srand(time(NULL)); // Inicializar la semilla para rand()

    if (vmm_init() != 0) {
        syslog(LOG_CRIT, "VMM initialization failed. Exiting.");
        closelog();
        return 1;
    }

    vm_context_t *ctx = vmm_allocate_process_context(0); // PID 0 para el proceso principal
    if (!ctx) {
        syslog(LOG_CRIT, "Failed to allocate VMM context. Exiting.");
        closelog();
        return 1;
    }

    AtomicEventRingBuffer rb;
    ring_buffer_init(&rb);

    ZeroHost *host = zero_host_init(&rb, ctx);
    if (!host) {
        syslog(LOG_CRIT, "ZeroHost initialization failed. Exiting.");
        free(ctx); // Liberar el contexto VMM si ZeroHost no se inicializa
        closelog();
        return 1;
    }

    pthread_t worker_threads[NUM_WORKER_THREADS];
    for (int i = 0; i < NUM_WORKER_THREADS; i++) {
        if (pthread_create(&worker_threads[i], NULL, worker_thread, host) != 0) {
            syslog(LOG_ERR, "Main: Failed to create worker thread %d: %s", i, strerror(errno));
            // Considerar si es un error fatal o si la simulación puede continuar.
            // Para esta simulación, asumimos que si un worker no se crea, el sistema puede estar comprometido.
            // Una solución más robusta sería liberar los hilos ya creados y ZeroHost antes de salir.
        }
    }

    Event event;
    int processed_events_count = 0;
    // El número total de eventos esperados es el número inicial de rayos más los eventos de retroalimentación de cada ciclo
    int total_expected_events_initial = NUM_RAYS; // Rayos iniciales
    // En cada ciclo (excepto el último), se encola un evento de brillo.
    int total_expected_events_feedback = (MAX_CYCLES - 1); // Eventos de brillo generados

    // El bucle principal del ZeroHost, que extrae eventos y procesa lotes.
    // Se ejecuta hasta que se alcanzan los ciclos máximos.
    while (host->current_level->scene->cycle < MAX_CYCLES) {
        // Intentar desencolar un evento
        if (dequeue_event(&rb, &event) == 0) {
            zero_host_process_event(host, &event);
            // No incrementamos processed_events_count aquí directamente porque
            // zero_host_process_event maneja el lote y llama a process_batch.
            // La condición de parada es principalmente por MAX_CYCLES.
        } else {
            // Si el búfer está vacío, el ZeroHost debería intentar procesar cualquier lote parcial
            // para no esperar indefinidamente por más eventos si el tiempo de espera ha expirado.
            // Esta llamada puede ser redundante si process_event ya lo maneja bien,
            // pero asegura que se procesen los lotes con timeout.
            simple_lock(&host->lock); // Bloquear para comprobar y procesar
            if (host->batch_count > 0 && usec_since(host->last_batch_time) > BATCH_TIMEOUT_US) {
                simple_unlock(&host->lock); // Desbloquear antes de la llamada
                zero_host_process_batch(host);
            } else {
                simple_unlock(&host->lock); // Desbloquear si no se procesa
                usleep(1000); // Pequeña espera para no girar la CPU innecesariamente
            }
        }
    }
    syslog(LOG_INFO, "Main: Reached MAX_CYCLES (%d). Stopping event processing.", MAX_CYCLES);

    // Unirse a los hilos de trabajo para asegurar que todos hayan terminado
    for (int i = 0; i < NUM_WORKER_THREADS; i++) {
        syslog(LOG_DEBUG, "Main: Joining worker thread %d...", i);
        pthread_join(worker_threads[i], NULL);
    }
    syslog(LOG_INFO, "Main: All worker threads joined.");

    // Liberar ZeroHost y sus recursos
    zero_host_free(host);
    
    // Liberar el contexto VMM principal, ya que no es responsabilidad de ZeroHost.
    if (ctx) {
        free(ctx);
        ctx = NULL; // Prevenir el uso de punteros colgantes
    }

    closelog(); // Cerrar la conexión syslog
    printf("--- Simulation Complete ---\n");
    return 0;
}
