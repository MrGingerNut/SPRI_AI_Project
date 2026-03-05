#ifndef WILDFIRE_WEIGHTS_H
#define WILDFIRE_WEIGHTS_H

// Estadísticas de normalización
extern const float wildfire_norm_mean[4];
extern const float wildfire_norm_std[4];

// Número total de capas
#define WILDFIRE_NUM_LAYERS 76

// Estructura para información de capas
typedef struct {
    const char* name;
    const float* data;
    int shape[4];
    int ndim;
    int size;
} WildfireLayer;

// Declaraciones externas
#ifdef __cplusplus
extern "C" {
#endif

// Cargar pesos
int wildfire_load_weights(const char* base_path);

// Obtener capa por índice
const float* wildfire_get_layer_data(int layer_index, int* out_size);

// Obtener capa por nombre
const float* wildfire_get_layer_by_name(const char* name, int* out_size);

// Liberar memoria
void wildfire_free_weights(void);

#ifdef __cplusplus
}
#endif

#endif // WILDFIRE_WEIGHTS_H
